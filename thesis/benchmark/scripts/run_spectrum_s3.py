#!/usr/bin/env python
"""
Unified Spectrum Runner — KE (500 hPa), KE (850 hPa), or Q spectrum for S3 Aurora Predictions.

Computes spectra at multiple lead times for a single model, averaged over
biweekly init dates. Results are flushed incrementally to CSV.

Usage:
    python run_spectrum_s3.py ke --year 2022

    python run_spectrum_s3.py ke_850hpa --year 2022

    python run_spectrum_s3.py q --year 2022
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import boto3
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

# Add parent for physics_metrics import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_metrics import (
    compute_ke_spectrum,
    compute_q_spectrum,
    _detect_level_dim,
    U_NAMES,
    V_NAMES,
    Q_NAMES,
)

# ============================================================================
# Analysis configuration
# ============================================================================

# S3 Configuration
S3_BUCKET = "ekasteleyn-aurora-predictions"
S3_FOLDER = "aurora_hres_validation"
S3_ENDPOINT = "https://ceph-gw.science.uva.nl:8000"

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
DEFAULT_LEAD_HOURS = [6, 24, 48, 72, 120, 168, 240]
RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

VAR_NAME_MAP = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}

@dataclass(frozen=True)
class AnalysisCfg:
    """Encapsulates the differences between ke / ke_850hpa / q analyses."""
    name: str                                          # human-readable banner name
    csv_prefix: str                                    # output filename prefix
    default_level: float                               # default pressure level (hPa)
    value_col: str                                     # CSV column for spectral values
    compute_fn: Callable[..., tuple[np.ndarray, np.ndarray]]  # spectrum function
    needed_vars: tuple[str, ...]                       # variable names to subset lazily


ANALYSES: dict[str, AnalysisCfg] = {
    "ke": AnalysisCfg(
        name="KE SPECTRUM",
        csv_prefix="ke_spectrum",
        default_level=500.0,
        value_col="energy",
        compute_fn=compute_ke_spectrum,
        needed_vars=U_NAMES + V_NAMES,
    ),
    "ke_850hpa": AnalysisCfg(
        name="KE SPECTRUM (850 hPa)",
        csv_prefix="ke_spectrum_850hpa",
        default_level=850.0,
        value_col="energy",
        compute_fn=compute_ke_spectrum,
        needed_vars=U_NAMES + V_NAMES,
    ),
    "q": AnalysisCfg(
        name="Q SPECTRUM",
        csv_prefix="q_spectrum",
        default_level=500.0,
        value_col="power",
        compute_fn=compute_q_spectrum,
        needed_vars=Q_NAMES,
    ),
}

# ============================================================================
# S3 Utilities
# ============================================================================

def get_s3_client():
    """Initialize S3 client with credentials from .env file."""
    load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/scripts/steering/.env")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("UVA_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("UVA_S3_SECRET_KEY"),
    )

def get_s3_key(date_str: str, init_hour: int, lead_hours: int) -> str:
    """Construct S3 key for a specific prediction file.
    File format: aurora_pred_YYYYMMDD_HHMM_stepXX_XXXh.nc
    """
    date_fmt = date_str.replace("-", "")
    step = lead_hours // 6
    return f"{S3_FOLDER}/aurora_pred_{date_fmt}_{init_hour:02d}00_step{step:02d}_{lead_hours:03d}h.nc"

def download_prediction(s3_client, date_str: str, init_hour: int, lead_hours: int, 
                       local_dir: Path) -> Optional[Path]:
    """Download a prediction file from S3."""
    s3_key = get_s3_key(date_str, init_hour, lead_hours)
    local_path = local_dir / Path(s3_key).name
    
    try:
        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {s3_key}: {e}")

def load_prediction_from_s3(s3_client, date_str: str, init_hour: int, 
                           lead_hours: int, local_dir: Path) -> Optional[xr.Dataset]:
    """Load a prediction dataset from S3."""
    local_path = download_prediction(s3_client, date_str, init_hour, lead_hours, local_dir)
    ds = xr.open_dataset(local_path)
    
    # Rename variables to match WB2 naming convention
    rename_map = {k: v for k, v in VAR_NAME_MAP.items() if k in ds.data_vars}
    if rename_map:
        ds = ds.rename(rename_map)
    return ds

# ============================================================================
# Zarr helpers
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    rename = {}
    for v in list(ds.data_vars) + list(ds.dims):
        if v != v.strip():
            rename[v] = v.strip()
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds

def _align_era5_to_pred(
    ds_era5: xr.Dataset,
    ds_pred: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """
    Align ERA5 grid to match prediction grid — no interpolation.
    """
    n_era5 = ds_era5.sizes.get(lat_name, 0)
    n_pred = ds_pred.sizes.get(lat_name, 0)
    n_lon_era5 = ds_era5.sizes.get(lon_name, 0)
    n_lon_pred = ds_pred.sizes.get(lon_name, 0)

    if n_lon_era5 != n_lon_pred:
        raise ValueError(
            f"Longitude grid mismatch: ERA5 has {n_lon_era5}, "
            f"prediction has {n_lon_pred}. Cannot align without interpolation."
        )

    if n_era5 == n_pred:
        result = ds_era5
    elif n_era5 == n_pred + 1:
        lats = ds_era5[lat_name].values
        if lats[0] > lats[-1]:  # Descending (N→S): drop last row (south pole)
            result = ds_era5.isel({lat_name: slice(0, -1)})
        else:                   # Ascending (S→N): drop first row (south pole)
            result = ds_era5.isel({lat_name: slice(1, None)})
    else:
        raise ValueError(
            f"Latitude grid mismatch: ERA5 has {n_era5} rows, "
            f"prediction has {n_pred}. Only exact match or 1-row "
            f"pole difference is supported (no interpolation)."
        )

    result = result.assign_coords({lat_name: ds_pred[lat_name].values})
    
    if lat_name in result.dims and lon_name in result.dims:
        dims_list = list(result.dims)
        idx_lon = dims_list.index(lon_name)
        idx_lat = dims_list.index(lat_name)
        if idx_lon < idx_lat:
            dims_list[idx_lon], dims_list[idx_lat] = dims_list[idx_lat], dims_list[idx_lon]
            result = result.transpose(*dims_list)

    return result

def _lazy_subset(ds: xr.Dataset, needed_vars: tuple[str, ...]) -> xr.Dataset:
    """Subset dataset to only the variables present from needed_vars (lazy)."""
    keep = [v for v in ds.data_vars if v in needed_vars]
    if not keep:
        raise ValueError(
            f"None of the needed variables {needed_vars} found in "
            f"dataset. Available: {list(ds.data_vars)}"
        )
    return ds[keep]

# ============================================================================
# Core computation
# ============================================================================

def compute_spectra_for_model(
    cfg: AnalysisCfg,
    era5_zarr: str,
    year: int,
    model: str,
    lead_hours: list[int],
    level: float | None = None,
    max_dates: int | None = None,
    verbose: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if level is None:
        level = cfg.default_level
    vcol = cfg.value_col

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {cfg.name}: {model} ({year}) (S3)")
        print(f"  Reference:  {era5_zarr}")
        print(f"  Level: {level} hPa | Lead times: {lead_hours}")
        if output_path:
            print(f"  Incremental output: {output_path}")
        print(f"{'='*70}\n")

    ds_era5_full = open_zarr_anonymous(era5_zarr)
    s3_client = get_s3_client()

    init_dates = []
    for month in range(1, 13):
        for day in [1, 15]:
            try:
                init_dates.append(np.datetime64(f"{year}-{month:02d}-{day:02d}"))
            except ValueError:
                pass
    if max_dates:
        init_dates = init_dates[:max_dates]

    if verbose:
        print(f"  Init dates: {len(init_dates)}")

    rows: list[dict] = []
    header_written = output_path.exists() if output_path else False

    with tempfile.TemporaryDirectory() as tmpdir_path:
        tmpdir = Path(tmpdir_path)
        for i, init_time in enumerate(init_dates, 1):
            if verbose:
                print(f"\n  [{i}/{len(init_dates)}] {init_time}")

            date_rows: list[dict] = []
            date_str = str(init_time).split("T")[0]
            init_hour = 0

            for lead_h in lead_hours:
                lead_td = np.timedelta64(lead_h, "h")
                valid_time = init_time + lead_td

                ds_snap = None
                try:
                    ds_snap = load_prediction_from_s3(s3_client, date_str, init_hour, lead_h, tmpdir).load()
                    ds_snap = _lazy_subset(ds_snap, cfg.needed_vars)
                    
                    level_dim = _detect_level_dim(ds_snap)

                    k_pred, v_pred = cfg.compute_fn(ds_snap, level=level, level_dim=level_dim)

                    for wi in range(len(k_pred)):
                        date_rows.append({
                            "model": model,
                            "lead_hours": lead_h,
                            "wavenumber": int(k_pred[wi]),
                            vcol: float(v_pred[wi]),
                            "source": "pred",
                            "date": str(init_time),
                        })

                    if verbose:
                        print(f"    {lead_h:>4d}h pred ✓ (max_l={len(k_pred)-1})", end="")

                except Exception as exc:
                    if verbose:
                        print(f"    {lead_h:>4d}h pred ✗ ({exc})", end="")
                    continue  # If we don't have prediction, we can skip ERA5
                finally:
                    if ds_snap is not None:
                        ds_snap.close()

                try:
                    ds_era5_snap = ds_era5_full.sel(time=valid_time, method="nearest")
                    
                    # We need another query to get grid match
                    ds_pred_dummy = load_prediction_from_s3(s3_client, date_str, init_hour, lead_h, tmpdir)
                    ds_era5_snap = _align_era5_to_pred(ds_era5_snap, ds_pred_dummy)
                    ds_pred_dummy.close()

                    ds_era5_snap = _lazy_subset(ds_era5_snap, cfg.needed_vars).load()
                    
                    # ERA5 level dim
                    level_dim_era5 = _detect_level_dim(ds_era5_snap)

                    k_era5, v_era5 = cfg.compute_fn(ds_era5_snap, level=level, level_dim=level_dim_era5)

                    for wi in range(len(k_era5)):
                        date_rows.append({
                            "model": model,
                            "lead_hours": lead_h,
                            "wavenumber": int(k_era5[wi]),
                            vcol: float(v_era5[wi]),
                            "source": "era5",
                            "date": str(init_time),
                        })

                    if verbose:
                        print(f" | era5 ✓")
                    del ds_era5_snap

                except Exception as exc:
                    if verbose:
                        print(f" | era5 ✗ ({exc})")

            # Flush to disk after each init date
            rows.extend(date_rows)
            if output_path and date_rows:
                df_chunk = pd.DataFrame(date_rows)
                df_chunk.to_csv(
                    output_path,
                    mode="a",
                    header=not header_written,
                    index=False,
                )
                header_written = True
                if verbose:
                    print(f"    [flushed {len(date_rows)} rows to {output_path.name}]")

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute spectral analysis (KE or Q) at multiple lead times for Aurora S3"
    )
    parser.add_argument(
        "analysis", choices=list(ANALYSES),
        help="Analysis type: ke (500 hPa), ke_850hpa (850 hPa), or q (humidity)",
    )
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--era5-zarr", type=str, default=ERA5_ZARR)
    parser.add_argument("--level", type=float, default=None,
                        help="Override pressure level (hPa); default depends on analysis")
    parser.add_argument("--lead-hours", type=str, default=None,
                        help="Comma-separated lead hours (default: 6,24,48,72,120,168,240)")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Limit number of init dates (for testing)")
    parser.add_argument("--append", action="store_true", help="Append to existing output CSV")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = ANALYSES[args.analysis]

    lead_hours = (
        [int(h) for h in args.lead_hours.split(",")]
        if args.lead_hours
        else DEFAULT_LEAD_HOURS
    )

    output = Path(args.output) if args.output else (
        RESULTS_DIR / f"{cfg.csv_prefix}_aurora_s3_{args.year}.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        output.unlink()

    df = compute_spectra_for_model(
        cfg=cfg,
        era5_zarr=args.era5_zarr,
        year=args.year,
        model="aurora_s3",
        lead_hours=lead_hours,
        level=args.level,
        max_dates=args.max_dates,
        verbose=not args.quiet,
        output_path=output,
    )

    print(f"\n✓ Saved {len(df)} rows → {output}")
    if len(df) > 0:
        print(f"  Unique lead times: {sorted(df['lead_hours'].unique())}")
        print(f"  Unique dates: {df['date'].nunique()}")

if __name__ == "__main__":
    main()
