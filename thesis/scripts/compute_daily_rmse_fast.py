#!/usr/bin/env python3
"""
Fast daily RMSE computation using batch loading.

Loads all data at once per model, then computes RMSE vectorized.
Much faster than per-date loading (~minutes instead of hours).

Usage:
    python compute_daily_rmse_fast.py --model pangu
    python compute_daily_rmse_fast.py --model all
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

# Model configurations
MODEL_CONFIG = {
    "pangu": {
        "pred_zarr": "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "fuxi": {
        "pred_zarr": "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "graphcast": {
        "pred_zarr": "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "hres": {
        "pred_zarr": "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    },
    "neuralgcm": {
        "pred_zarr": "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr",
        "era5_zarr": "gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr",
    },
}

TARGET_LEAD_HOURS = [12, 120, 240]


def compute_rmse_batch(model: str, year: int = 2020) -> pd.DataFrame:
    """Compute daily RMSE for all dates and lead times at once."""
    config = MODEL_CONFIG[model]
    
    print(f"  Opening prediction Zarr (one-time load)...")
    ds_pred = xr.open_zarr(config["pred_zarr"], storage_options={"token": "anon"})
    
    print(f"  Opening ERA5 Zarr (one-time load)...")
    ds_era5 = xr.open_zarr(config["era5_zarr"], storage_options={"token": "anon"})
    
    # Normalize dimension names
    for ds in [ds_pred, ds_era5]:
        if "lat" in ds.dims:
            ds = ds.rename({"lat": "latitude"})
        if "lon" in ds.dims:
            ds = ds.rename({"lon": "longitude"})
    
    # Find dimensions
    pred_td_dim = next((d for d in ["prediction_timedelta", "lead_time", "step"] if d in ds_pred.dims), None)
    level_dim = next((d for d in ["level", "pressure_level"] if d in ds_pred.dims), None)
    
    # Get available times in prediction dataset for the year
    pred_times = pd.to_datetime(ds_pred.time.values)
    year_mask = pred_times.year == year
    year_times = pred_times[year_mask]
    
    print(f"  Found {len(year_times)} init times in {year}")
    
    # Get available lead times
    if pred_td_dim:
        avail_leads = ds_pred[pred_td_dim].values
        avail_leads_h = [int(td / np.timedelta64(1, 'h')) for td in avail_leads]
        print(f"  Available lead times: {avail_leads_h[:10]}... (total {len(avail_leads_h)})")
    
    results = []
    
    for lead_h in TARGET_LEAD_HOURS:
        print(f"\n  Processing lead_time={lead_h}h...")
        lead_td = np.timedelta64(lead_h, 'h')
        
        # Check if this lead time exists
        if pred_td_dim and lead_h not in avail_leads_h:
            # Find nearest
            nearest_idx = np.argmin(np.abs(np.array(avail_leads_h) - lead_h))
            actual_lead_h = avail_leads_h[nearest_idx]
            if abs(actual_lead_h - lead_h) > 6:
                print(f"    ⚠ Lead time {lead_h}h not available, skipping")
                continue
            lead_td = avail_leads[nearest_idx]
            print(f"    Using nearest lead time: {actual_lead_h}h")
        
        # Select prediction data for this lead time (all init times at once)
        print(f"    Loading predictions...")
        if pred_td_dim:
            ds_p = ds_pred.sel({pred_td_dim: lead_td}, method="nearest")
        else:
            ds_p = ds_pred
        
        # Select year
        ds_p = ds_p.sel(time=year_times)
        
        # Load Z500 and T850 predictions
        z_pred = ds_p["geopotential"].sel({level_dim: 500}) if level_dim else ds_p["geopotential"]
        t_pred = ds_p["temperature"].sel({level_dim: 850}) if level_dim else ds_p["temperature"]
        
        print(f"    Loading prediction arrays...")
        z_pred_vals = z_pred.load()  # Load into memory once
        t_pred_vals = t_pred.load()
        
        # Compute valid times
        valid_times = year_times + pd.Timedelta(hours=lead_h)
        
        # Load ERA5 at valid times
        print(f"    Loading ERA5 truth...")
        era5_level_dim = next((d for d in ["level", "pressure_level"] if d in ds_era5.dims), None)
        
        z_era5 = ds_era5["geopotential"].sel(time=valid_times.values)
        t_era5 = ds_era5["temperature"].sel(time=valid_times.values)
        
        if era5_level_dim:
            z_era5 = z_era5.sel({era5_level_dim: 500})
            t_era5 = t_era5.sel({era5_level_dim: 850})
        
        z_era5_vals = z_era5.load()
        t_era5_vals = t_era5.load()
        
        # Compute RMSE per date
        print(f"    Computing RMSE for {len(year_times)} dates...")
        for i, init_time in enumerate(year_times):
            date_str = init_time.strftime("%Y-%m-%d")
            
            # Get slices for this date
            z_p = z_pred_vals.sel(time=init_time).values
            t_p = t_pred_vals.sel(time=init_time).values
            z_e = z_era5_vals.isel(time=i).values
            t_e = t_era5_vals.isel(time=i).values
            
            # Handle shape mismatch (interpolation would be better, but this is fast)
            if z_p.shape != z_e.shape:
                min_shape = tuple(min(a, b) for a, b in zip(z_p.shape, z_e.shape))
                z_p = z_p[:min_shape[0], :min_shape[1]]
                z_e = z_e[:min_shape[0], :min_shape[1]]
                t_p = t_p[:min_shape[0], :min_shape[1]]
                t_e = t_e[:min_shape[0], :min_shape[1]]
            
            z500_rmse = np.sqrt(np.nanmean((z_p - z_e) ** 2))
            t850_rmse = np.sqrt(np.nanmean((t_p - t_e) ** 2))
            
            results.append({
                "date": date_str,
                "lead_time_hours": lead_h,
                "z500_rmse": z500_rmse,
                "t850_rmse": t850_rmse,
            })
        
        print(f"    ✓ Done")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pangu", 
                        help="Model name or 'all'")
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()
    
    models = list(MODEL_CONFIG.keys()) if args.model == "all" else [args.model]
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"  {model.upper()}")
        print(f"{'='*60}")
        
        try:
            df = compute_rmse_batch(model, args.year)
            
            # Save
            out_path = RESULTS_DIR / f"daily_rmse_{model}_{args.year}.csv"
            df.to_csv(out_path, index=False)
            print(f"\n  ✓ Saved {len(df)} rows to {out_path}")
            
            # Show sample
            print(f"\n  Sample:")
            print(df.head(6).to_string(index=False))
            
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
