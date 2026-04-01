#!/usr/bin/env python
"""
Physics Evaluation Runner — S3 Aurora Predictions

Loads Aurora predictions from S3 (individual NetCDF files) and computes
all physics metrics at multiple forecast horizons, saving results to CSV.

This is a minimal wrapper around run_physics_evaluation.py that handles
the S3 data loading.

Usage
-----
  python run_physics_evaluation_s3.py --dates 2022-01-01 2022-01-02
  python run_physics_evaluation_s3.py --year 2022 --workers 16
"""

from __future__ import annotations

import argparse
import calendar
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

# Import physics metrics from companion library
import sys
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import (
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_pure_tcwv,
    _find_effective_resolution,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_spectral_scores,
    derive_surface_pressure,
    get_grid_cell_area,
    _find_var,
    _detect_level_dim,
    SP_NAMES,
    MSL_NAMES,
    T_NAMES,
    T2M_NAMES,
    ZSFC_NAMES,
    Q_NAMES,
    U_NAMES,
    V_NAMES,
    PHI_NAMES,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

# S3 Configuration
S3_BUCKET = "ekasteleyn-aurora-predictions"
S3_FOLDER = "aurora_hres_validation"
S3_ENDPOINT = "https://ceph-gw.science.uva.nl:8000"

# ERA5 ground truth
ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
ERA5_DAILY_ZARR = "gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"

OUTPUT_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

# Target lead times
LEAD_TIMES: list[tuple[str, int]] = [
    ("12h", 12),
    ("24h", 24),
    ("5d", 120),
    ("10d", 240),
]

DEFAULT_WORKERS = 4

# Variable name mappings (S3 files use short names)
VAR_NAME_MAP = {
    "t": "temperature",
    "z": "geopotential", 
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
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
        return None


def load_prediction_from_s3(s3_client, date_str: str, init_hour: int, 
                           lead_hours: int, local_dir: Path) -> Optional[xr.Dataset]:
    """Load a prediction dataset from S3."""
    local_path = download_prediction(s3_client, date_str, init_hour, lead_hours, local_dir)
    if local_path is None:
        return None
    
    ds = xr.open_dataset(local_path)
    
    # Rename variables to match WB2 naming convention
    rename_map = {k: v for k, v in VAR_NAME_MAP.items() if k in ds.data_vars}
    if rename_map:
        ds = ds.rename(rename_map)
    
    return ds


# ============================================================================
# Zarr I/O (for ERA5)
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    # Normalise dimension names
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_static_fields(ds_era5: xr.Dataset) -> xr.Dataset:
    """Extract static fields from ERA5."""
    static_vars = {}

    def _extract_static(ds, name):
        var = ds[name]
        if "time" in var.dims:
            var = var.isel(time=0, drop=True)
        return var

    for name in ("geopotential_at_surface", "z_sfc", "orography"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    for name in ("land_sea_mask", "lsm"):
        if name in ds_era5.data_vars:
            static_vars[name] = _extract_static(ds_era5, name)
            break

    return xr.Dataset(static_vars)


def _get_ps(ds: xr.Dataset, ds_static: xr.Dataset, level_dim: str = "level",
           t2m_mean: Optional[xr.DataArray] = None) -> xr.DataArray:
    """Return surface pressure, deriving from MSL if needed."""
    sp_name = _find_var(ds, SP_NAMES)
    if sp_name is not None:
        sp = ds[sp_name]
        sp.attrs["derivation_method"] = "direct_sp"
        return sp

    try:
        sp = derive_surface_pressure(ds, ds_static, t2m_mean=t2m_mean)
        sp.attrs["derivation_method"] = "hypsometric_msl"
        return sp
    except (ValueError, KeyError):
        pass

    raise ValueError("Cannot derive surface pressure")


# ============================================================================
# Single-Slice Evaluation
# ============================================================================

def _evaluate_one(
    date_str: str,
    init_hour: int,
    lead_hours: int,
    counter: int,
    total: int,
    verbose: bool,
) -> list[dict]:
    """
    Evaluate physics metrics for one (date, init_hour, lead_time) combination.
    """
    rows = []
    
    def _log(msg):
        if verbose:
            print(msg, flush=True)
    
    _log(f"  [{counter}/{total}] {date_str} init={init_hour:02d}:00 lead=+{lead_hours}h")
    
    try:
        # Initialize S3 client in worker process
        s3_client = get_s3_client()
        
        # Create temp directory for downloads
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Load prediction from S3
            ds_aurora = load_prediction_from_s3(
                s3_client, date_str, init_hour, lead_hours, tmpdir
            )
            
            if ds_aurora is None:
                _log(f"    ⚠ File not found on S3")
                return rows
            
            # Load ERA5
            ds_era5 = open_zarr_anonymous(ERA5_ZARR)
            ds_era5_daily = open_zarr_anonymous(ERA5_DAILY_ZARR)
            ds_static = load_static_fields(ds_era5)
            
            # Compute valid time
            init_time = np.datetime64(f"{date_str}T{init_hour:02d}:00", "ns")
            lead_td = np.timedelta64(lead_hours, "h")
            valid_time = init_time + lead_td
            
            # Grid cell area
            area = get_grid_cell_area(ds_era5.isel(time=0, drop=True))
            
            # Get static fields
            z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
            z_sfc = ds_static[z_sfc_name] if z_sfc_name else None
            
            # Level dimension
            level_dim = _detect_level_dim(ds_aurora)
            n_levels = len(ds_aurora[level_dim]) if level_dim else None
            
            def _append(metric_name, model_val, era5_val=None):
                rows.append({
                    "date": date_str,
                    "init_hour": init_hour,
                    "lead_time_hours": lead_hours,
                    "metric_name": metric_name,
                    "model_value": model_val,
                    "era5_value": era5_val,
                    "n_levels": n_levels,
                })
            
            # Load Aurora data
            ds_aurora = ds_aurora.load()
            
            # Load ERA5 at valid time
            ds_era5_t = ds_era5.sel(time=valid_time, method="nearest").load()
            
            # --- Compute Metrics ---
            
            # 1. Hydrostatic Imbalance
            try:
                hydro_aurora = compute_hydrostatic_imbalance(ds_aurora, level_dim=level_dim)
                hydro_era5 = compute_hydrostatic_imbalance(ds_era5_t)
                _append("hydrostatic_rmse", float(hydro_aurora), float(hydro_era5))
                _log(f"    Hydrostatic RMSE: {hydro_aurora:.4f} (ERA5: {hydro_era5:.4f})")
            except Exception as e:
                _log(f"    ⚠ Hydrostatic error: {e}")
            
            # 2. Geostrophic Imbalance
            try:
                geo_aurora = compute_geostrophic_imbalance(ds_aurora, level_dim=level_dim)
                geo_era5 = compute_geostrophic_imbalance(ds_era5_t)
                _append("geostrophic_rmse", float(geo_aurora), float(geo_era5))
                _log(f"    Geostrophic RMSE: {geo_aurora:.4f} (ERA5: {geo_era5:.4f})")
            except Exception as e:
                _log(f"    ⚠ Geostrophic error: {e}")
            
            # 3. Conservation metrics (mass, water, energy)
            try:
                # Get surface pressure
                ps_aurora = _get_ps(ds_aurora, ds_static, level_dim=level_dim)
                ps_era5 = _get_ps(ds_era5_t, ds_static)
                
                cons_aurora = compute_conservation_scalars(
                    ds_aurora, ps_aurora, area, level_dim=level_dim
                )
                cons_era5 = compute_conservation_scalars(ds_era5_t, ps_era5, area)
                
                for key in ["dry_mass_Eg", "water_mass_kg", "total_energy_J"]:
                    if key in cons_aurora and key in cons_era5:
                        _append(key, cons_aurora[key], cons_era5[key])
                        _log(f"    {key}: {cons_aurora[key]:.4e} (ERA5: {cons_era5[key]:.4e})")
            except Exception as e:
                _log(f"    ⚠ Conservation error: {e}")
            
            # 4. KE Spectrum
            try:
                spec_aurora = compute_ke_spectrum(ds_aurora, level_dim=level_dim)
                spec_era5 = compute_ke_spectrum(ds_era5_t)
                
                if spec_aurora is not None and spec_era5 is not None:
                    scores = compute_spectral_scores(spec_aurora, spec_era5)
                    for score_name, score_val in scores.items():
                        _append(f"spectrum_{score_name}", score_val)
                    _log(f"    Spectrum L1: {scores.get('l1_error', 'N/A'):.4f}")
            except Exception as e:
                _log(f"    ⚠ Spectrum error: {e}")
            
            ds_aurora.close()
    
    except Exception as e:
        _log(f"    ⚠ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return rows


# ============================================================================
# Main Runner
# ============================================================================

def get_dates_from_args(args) -> list[str]:
    """Parse date arguments."""
    if args.dates:
        return args.dates
    if args.month:
        year, month = args.month.split("-")
        n_days = calendar.monthrange(int(year), int(month))[1]
        return [f"{year}-{month}-{d:02d}" for d in range(1, n_days + 1)]
    year = args.year
    dates = []
    for m in range(1, 13):
        n_days = calendar.monthrange(year, m)[1]
        for d in range(1, n_days + 1):
            dates.append(f"{year}-{m:02d}-{d:02d}")
    return dates


def run_evaluation(
    dates: list[str],
    output_csv: Path,
    init_hour: int = 0,
    lead_times: list[tuple[str, int]] = None,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run physics evaluation for Aurora S3 predictions.
    """
    _lead_times = lead_times if lead_times is not None else LEAD_TIMES
    
    # Build work items
    work_items = [
        (date_str, init_hour, lead_hours, lead_label)
        for date_str in dates
        for lead_label, lead_hours in _lead_times
    ]
    n_total = len(work_items)
    
    if verbose:
        print("\n" + "=" * 70)
        print("  PHYSICS EVALUATION — S3 Aurora Predictions")
        print("=" * 70)
        print(f"  S3 Bucket  : s3://{S3_BUCKET}/{S3_FOLDER}")
        print(f"  ERA5       : {ERA5_ZARR}")
        print(f"  Dates      : {len(dates)}")
        print(f"  Init hour  : {init_hour:02d}:00")
        print(f"  Lead times : {[label for label, _ in _lead_times]}")
        print(f"  Total evals: {n_total}")
        print(f"  Workers    : {workers}")
        print(f"  Output     : {output_csv}")
        print("=" * 70)
    
    all_rows = []
    
    if workers == 1:
        # Sequential execution
        for idx, (date_str, init_hr, lead_hours, _) in enumerate(work_items, 1):
            rows = _evaluate_one(date_str, init_hr, lead_hours, idx, n_total, verbose)
            all_rows.extend(rows)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for idx, (date_str, init_hr, lead_hours, _) in enumerate(work_items, 1):
                fut = pool.submit(
                    _evaluate_one, date_str, init_hr, lead_hours, idx, n_total, verbose
                )
                futures[fut] = (date_str, lead_hours)
            
            for fut in as_completed(futures):
                try:
                    rows = fut.result(timeout=600)
                    all_rows.extend(rows)
                except Exception as e:
                    date_str, lead_hours = futures[fut]
                    print(f"  ⚠ Failed {date_str} +{lead_hours}h: {e}")
    
    # Save results
    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"\n✓ Saved {len(df)} rows to {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Physics evaluation for Aurora S3 predictions"
    )
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=str, default=None,
                       help="Single month (YYYY-MM)")
    parser.add_argument("--dates", nargs="+", type=str, default=None,
                       help="Specific dates (YYYY-MM-DD)")
    parser.add_argument("--init-hour", type=int, default=0,
                       help="Initialization hour (0 or 12)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    dates = get_dates_from_args(args)
    
    output_csv = (
        Path(args.output) if args.output 
        else OUTPUT_DIR / f"physics_aurora_s3_{args.year}.csv"
    )
    
    run_evaluation(
        dates=dates,
        output_csv=output_csv,
        init_hour=args.init_hour,
        workers=args.workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
