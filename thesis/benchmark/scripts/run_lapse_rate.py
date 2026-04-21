#!/usr/bin/env python
"""
Lapse Rate Evaluation Runner.

Computes the Lapse Rate Wasserstein Distance for a single model against ERA5,
evaluating the thermodynamic profile distribution at multiple lead times.
Results are incrementally flushed to CSV.

Usage:
    python run_lapse_rate.py --model aurora --year 2022 \
        --prediction-zarr gs://weatherbench2/datasets/aurora/2022-1440x721.zarr

    python run_lapse_rate.py --model pangu --year 2020 \
        --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Add parent for physics_metrics import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_metrics import (
    compute_lapse_rate_wasserstein,
    get_grid_cell_area,
    _detect_pred_td_dim,
    T_NAMES,
    PHI_NAMES,
)

# ============================================================================
# Configuration
# ============================================================================

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
DEFAULT_LEAD_HOURS = [12, 24, 48, 72, 120, 168, 240]
RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "benchmark" / "results"

def open_zarr_anonymous(url: str) -> xr.Dataset:
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

def _lazy_subset(ds: xr.Dataset, needed_vars: tuple[str, ...]) -> xr.Dataset:
    keep = [v for v in ds.data_vars if v in needed_vars]
    return ds[keep]

def compute_lapse_rates(
    prediction_zarr: str,
    era5_zarr: str,
    year: int,
    model: str,
    lead_hours: list[int],
    max_dates: int | None = None,
    verbose: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  LAPSE RATE W1 EVALUATION: {model} ({year})")
        print(f"  Prediction: {prediction_zarr}")
        print(f"  Reference:  {era5_zarr}")
        print(f"  Lead times: {lead_hours}")
        if output_path:
            print(f"  Output: {output_path}")
        print(f"{'='*70}\n")

    ds_pred_full = open_zarr_anonymous(prediction_zarr)
    ds_era5_full = open_zarr_anonymous(era5_zarr)

    pred_td_dim = _detect_pred_td_dim(ds_pred_full)
    has_pred_td = pred_td_dim is not None

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

    # Pre-calculate area from ERA5
    # Warning: assumes Prediction and ERA5 grids are exactly equivalent.
    area = get_grid_cell_area(ds_era5_full.isel(time=0, drop=True))
    needed_vars = tuple(T_NAMES) + tuple(PHI_NAMES)
    
    rows: list[dict] = []
    header_written = output_path.exists() if output_path else False

    for i, init_time in enumerate(init_dates, 1):
        if verbose:
            print(f"\n  [{i}/{len(init_dates)}] {init_time}")

        date_rows: list[dict] = []

        for lead_h in lead_hours:
            lead_td = np.timedelta64(lead_h, "h")
            valid_time = init_time + lead_td

            try:
                if has_pred_td:
                    ds_snap = ds_pred_full.sel(time=init_time).sel({pred_td_dim: lead_td})
                else:
                    ds_snap = ds_pred_full.sel(time=valid_time)

                ens_dims = [d for d in ("number", "realization", "sample") if d in ds_snap.dims]
                if ens_dims:
                    if ds_snap.sizes[ens_dims[0]] > 10:
                        ds_snap = ds_snap.mean(dim=ens_dims[0])
                    else:
                        ds_snap = ds_snap.isel({ens_dims[0]: 0})

                ds_snap = _lazy_subset(ds_snap, needed_vars).load()
                
                ds_era5_snap = ds_era5_full.sel(time=valid_time)
                ds_era5_snap = _lazy_subset(ds_era5_snap, needed_vars).load()

                res_dict = compute_lapse_rate_wasserstein(ds_snap, ds_era5_snap, area)
                
                for band_key, w1_val in res_dict.items():
                    date_rows.append({
                        "model": model,
                        "lead_hours": lead_h,
                        "metric_name": band_key,
                        "value": w1_val,
                        "date": str(init_time),
                    })

                if verbose:
                    print(f"    {lead_h:>4d}h ✓")

            except Exception as exc:
                if verbose:
                    print(f"    {lead_h:>4d}h ✗ ({exc})")

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

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute Lapse Rate Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--prediction-zarr", type=str, required=True)
    parser.add_argument("--era5-zarr", type=str, default=ERA5_ZARR)
    parser.add_argument("--lead-hours", type=str, default=None,
                        help="Comma-separated lead hours (default: 12,24,48,72,120,168,240)")
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    lead_hours = (
        [int(h) for h in args.lead_hours.split(",")]
        if args.lead_hours else DEFAULT_LEAD_HOURS
    )

    output = Path(args.output) if args.output else (
        RESULTS_DIR / f"lapse_rate_w1_{args.model}_{args.year}.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        output.unlink()

    df = compute_lapse_rates(
        prediction_zarr=args.prediction_zarr,
        era5_zarr=args.era5_zarr,
        year=args.year,
        model=args.model,
        lead_hours=lead_hours,
        max_dates=args.max_dates,
        verbose=not args.quiet,
        output_path=output,
    )
    print(f"\n✓ Saved {len(df)} rows → {output}")

if __name__ == "__main__":
    main()
