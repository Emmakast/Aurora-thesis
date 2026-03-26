#!/usr/bin/env python
"""
Unified Spectrum Runner — KE (500 hPa), KE (850 hPa), or Q spectrum.

Computes spectra at multiple lead times for a single model, averaged over
biweekly init dates.  Results are flushed incrementally to CSV.

Usage:
    python run_spectrum.py ke --model aurora --year 2022 \
        --prediction-zarr gs://weatherbench2/datasets/aurora/2022-1440x721.zarr

    python run_spectrum.py ke_850hpa --model pangu --year 2020 \
        --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr

    python run_spectrum.py q --model graphcast --year 2020 \
        --prediction-zarr gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

# Add parent for physics_metrics import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_metrics import (
    compute_ke_spectrum,
    compute_q_spectrum,
    _detect_pred_td_dim,
    U_NAMES,
    V_NAMES,
    Q_NAMES,
)

# ============================================================================
# Analysis configuration
# ============================================================================

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
DEFAULT_LEAD_HOURS = [6, 24, 48, 72, 120, 168, 240]
RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"


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

    Only two cases are allowed:
    1. Exact shape match.
    2. ERA5 has exactly 1 extra latitude row (721 vs 720) — drop pole.
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
    prediction_zarr: str,
    era5_zarr: str,
    year: int,
    model: str,
    lead_hours: list[int],
    level: float | None = None,
    max_dates: int | None = None,
    verbose: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compute spectra for a model at multiple lead times.

    If *output_path* is given, results are flushed to disk after every init
    date so that partial progress survives job cancellation.

    Returns a DataFrame with columns:
        model, lead_hours, wavenumber, <value_col>, source (pred/era5), date
    """
    if level is None:
        level = cfg.default_level
    vcol = cfg.value_col

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {cfg.name}: {model} ({year})")
        print(f"  Prediction: {prediction_zarr}")
        print(f"  Reference:  {era5_zarr}")
        print(f"  Level: {level} hPa | Lead times: {lead_hours}")
        if output_path:
            print(f"  Incremental output: {output_path}")
        print(f"{'='*70}\n")

    ds_pred_full = open_zarr_anonymous(prediction_zarr)
    ds_era5_full = open_zarr_anonymous(era5_zarr)

    pred_td_dim = _detect_pred_td_dim(ds_pred_full)
    has_pred_td = pred_td_dim is not None

    if verbose:
        print(f"  Prediction vars: {list(ds_pred_full.data_vars)[:10]}")
        print(f"  Has prediction_timedelta: {has_pred_td} ({pred_td_dim})")

    # Biweekly init dates (1st and 15th of each month)
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
    header_written = False

    for i, init_time in enumerate(init_dates, 1):
        if verbose:
            print(f"\n  [{i}/{len(init_dates)}] {init_time}")

        date_rows: list[dict] = []

        for lead_h in lead_hours:
            lead_td = np.timedelta64(lead_h, "h")
            valid_time = init_time + lead_td

            # --- Prediction spectrum ---
            try:
                if has_pred_td:
                    ds_snap = ds_pred_full.sel(time=init_time).sel(
                        {pred_td_dim: lead_td}
                    )
                else:
                    ds_snap = ds_pred_full.sel(time=valid_time)

                ens_dims = [d for d in ("number", "realization", "sample")
                            if d in ds_snap.dims]
                if ens_dims:
                    raise ValueError(
                        "This script is for deterministic forecasts. "
                        "Please subset the ensemble first."
                    )

                ds_snap = _lazy_subset(ds_snap, cfg.needed_vars).load()
                k_pred, v_pred = cfg.compute_fn(ds_snap, level=level)

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
                del ds_snap

            except Exception as exc:
                if verbose:
                    print(f"    {lead_h:>4d}h pred ✗ ({exc})", end="")

            # --- ERA5 / reference spectrum ---
            try:
                ds_era5_snap = ds_era5_full.sel(time=valid_time)
                ds_era5_snap = _align_era5_to_pred(ds_era5_snap, ds_pred_full)
                ds_era5_snap = _lazy_subset(ds_era5_snap, cfg.needed_vars).load()
                k_era5, v_era5 = cfg.compute_fn(ds_era5_snap, level=level)

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
        description="Compute spectral analysis (KE or Q) at multiple lead times"
    )
    parser.add_argument(
        "analysis", choices=list(ANALYSES),
        help="Analysis type: ke (500 hPa), ke_850hpa (850 hPa), or q (humidity)",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (aurora, pangu, graphcast, ...)")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--prediction-zarr", type=str, required=True)
    parser.add_argument("--era5-zarr", type=str, default=ERA5_ZARR)
    parser.add_argument("--level", type=float, default=None,
                        help="Override pressure level (hPa); default depends on analysis")
    parser.add_argument("--lead-hours", type=str, default=None,
                        help="Comma-separated lead hours (default: 6,24,48,72,120,168,240)")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Limit number of init dates (for testing)")
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
        RESULTS_DIR / f"{cfg.csv_prefix}_{args.model}_{args.year}.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    df = compute_spectra_for_model(
        cfg=cfg,
        prediction_zarr=args.prediction_zarr,
        era5_zarr=args.era5_zarr,
        year=args.year,
        model=args.model,
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
