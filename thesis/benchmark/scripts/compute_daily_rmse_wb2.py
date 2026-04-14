#!/usr/bin/env python3
"""
Compute daily RMSE using WB2 240x121 datasets and official WB2 methodology.

- Uses 240x121 resolution for all models
- HRES compared against IFS HRES t=0 (analysis)
- All other models compared against ERA5
- Uses weatherbench2 library for latitude-weighted RMSE

Usage:
    python compute_daily_rmse_wb2.py --model hres
    python compute_daily_rmse_wb2.py --model graphcast
    python compute_daily_rmse_wb2.py --model all
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from weatherbench2 import metrics as wb2_metrics

warnings.filterwarnings("ignore")

RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

# 240x121 resolution datasets
MODEL_CONFIG = {
    "pangu": {
        "pred_zarr": "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr",
        "truth_zarr": "gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        "truth_type": "era5",
    },
    "fuxi": {
        "pred_zarr": "gs://weatherbench2/datasets/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr",
        "truth_zarr": "gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        "truth_type": "era5",
    },
    "graphcast": {
        "pred_zarr": "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr",
        "truth_zarr": "gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        "truth_type": "era5",
    },
    "hres": {
        "pred_zarr": "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr",
        "truth_zarr": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
        "truth_type": "hres_t0",
    },
    "hres_era5": {
        "pred_zarr": "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr",
        "truth_zarr": "gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        "truth_type": "era5",
    },
    "neuralgcm": {
        "pred_zarr": "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr",
        "truth_zarr": "gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        "truth_type": "era5",
    },
}

TARGET_LEAD_HOURS = [12, 120, 240]


def compute_wb2_rmse(forecast: xr.Dataset, truth: xr.Dataset) -> dict:
    """Compute RMSE using official WB2 methodology."""
    # Get latitude weights using WB2's function
    weights = wb2_metrics.get_lat_weights(forecast)
    
    results = {}
    for var in forecast.data_vars:
        if var not in truth.data_vars:
            continue
        diff_sq = (forecast[var] - truth[var]) ** 2
        # Weighted spatial mean, then sqrt
        weighted_mse = diff_sq.weighted(weights).mean(["latitude", "longitude"])
        results[var] = float(np.sqrt(weighted_mse.values))
    
    return results


def get_dim_name(ds, candidates):
    """Find dimension name from candidates."""
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    return None


def compute_rmse_for_model(model: str, year: int = 2020) -> pd.DataFrame:
    """Compute daily RMSE for a model using WB2 240x121 data."""
    config = MODEL_CONFIG[model]
    
    print(f"  Opening prediction Zarr: {config['pred_zarr']}")
    ds_pred = xr.open_zarr(config["pred_zarr"], storage_options={"token": "anon"})
    
    print(f"  Opening truth Zarr: {config['truth_zarr']}")
    ds_truth = xr.open_zarr(config["truth_zarr"], storage_options={"token": "anon"})
    
    # Normalize coordinate names
    if "lat" in ds_pred.dims:
        ds_pred = ds_pred.rename({"lat": "latitude", "lon": "longitude"})
    if "lat" in ds_truth.dims:
        ds_truth = ds_truth.rename({"lat": "latitude", "lon": "longitude"})
    
    # Find dimension names
    pred_td_dim = get_dim_name(ds_pred, ["prediction_timedelta", "lead_time", "step"])
    level_dim_pred = get_dim_name(ds_pred, ["level", "pressure_level"])
    level_dim_truth = get_dim_name(ds_truth, ["level", "pressure_level"])
    
    print(f"  Pred TD dim: {pred_td_dim}, Level dim: {level_dim_pred}")
    
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
            nearest_idx = np.argmin(np.abs(np.array(avail_leads_h) - lead_h))
            actual_lead_h = avail_leads_h[nearest_idx]
            if abs(actual_lead_h - lead_h) > 6:
                print(f"    ⚠ Lead time {lead_h}h not available, skipping")
                continue
            lead_td = avail_leads[nearest_idx]
            print(f"    Using nearest lead time: {actual_lead_h}h")
        
        # Process each date
        for i, init_time in enumerate(year_times):
            if i % 50 == 0:
                print(f"    Processing {i+1}/{len(year_times)}...")
            
            date_str = init_time.strftime("%Y-%m-%d")
            valid_time = init_time + pd.Timedelta(hours=lead_h)
            
            try:
                # Get prediction
                ds_p = ds_pred.sel(time=init_time)
                if pred_td_dim and pred_td_dim in ds_p.dims:
                    ds_p = ds_p.sel({pred_td_dim: lead_td}, method="nearest")
                
                # Get Z500 and T850 predictions
                z_pred = ds_p["geopotential"]
                t_pred = ds_p["temperature"]
                
                if level_dim_pred and level_dim_pred in z_pred.dims:
                    z_pred = z_pred.sel({level_dim_pred: 500})
                    t_pred = t_pred.sel({level_dim_pred: 850})
                
                # Get truth at valid time
                ds_t = ds_truth.sel(time=valid_time.to_datetime64(), method="nearest")
                
                z_truth = ds_t["geopotential"]
                t_truth = ds_t["temperature"]
                
                if level_dim_truth and level_dim_truth in z_truth.dims:
                    z_truth = z_truth.sel({level_dim_truth: 500})
                    t_truth = t_truth.sel({level_dim_truth: 850})
                
                # Create datasets for WB2 RMSE computation
                forecast_ds = xr.Dataset({
                    "z500": z_pred.load(),
                    "t850": t_pred.load(),
                })
                truth_ds = xr.Dataset({
                    "z500": z_truth.load(),
                    "t850": t_truth.load(),
                })
                
                # Compute RMSE using WB2 methodology
                rmse_results = compute_wb2_rmse(forecast_ds, truth_ds)
                
                results.append({
                    "date": date_str,
                    "lead_time_hours": lead_h,
                    "z500_rmse": rmse_results.get("z500", np.nan),
                    "t850_rmse": rmse_results.get("t850", np.nan),
                })
                
            except Exception as e:
                print(f"    ⚠ Error for {date_str} lead={lead_h}h: {e}")
                continue
        
        print(f"    ✓ Done with lead={lead_h}h")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Compute daily RMSE using WB2 240x121 datasets")
    parser.add_argument("--model", type=str, default="all", 
                        help="Model name or 'all'")
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()
    
    if args.model == "all":
        models = list(MODEL_CONFIG.keys())
    else:
        models = [args.model]
    
    for model in models:
        if model not in MODEL_CONFIG:
            print(f"Unknown model: {model}")
            continue
            
        print(f"\n{'='*60}")
        print(f"  {model.upper()}")
        print(f"{'='*60}")
        
        try:
            df = compute_rmse_for_model(model, args.year)
            
            if len(df) == 0:
                print(f"  ⚠ No results for {model}")
                continue
            
            # Save
            out_path = RESULTS_DIR / f"daily_rmse_{model}_{args.year}.csv"
            df.to_csv(out_path, index=False)
            print(f"\n  ✓ Saved {len(df)} rows to {out_path}")
            
            # Show summary
            print(f"\n  Summary (mean RMSE):")
            for lt in df["lead_time_hours"].unique():
                sub = df[df["lead_time_hours"] == lt]
                print(f"    Lead {lt}h: Z500={sub['z500_rmse'].mean():.2f} m²/s², T850={sub['t850_rmse'].mean():.2f} K")
            
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
