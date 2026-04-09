#!/usr/bin/env python3
"""
Evaluate Aurora 2022 predictions from S3 against ERA5 ground truth.

Downloads Aurora predictions from S3, computes RMSE/ACC against ERA5,
and outputs evaluation metrics by date, init time, and lead time.

Usage:
    python evaluate_aurora_2022.py
    python evaluate_aurora_2022.py --max-files 10  # Quick test
"""

import argparse
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import boto3
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ERA5 ground truth (WB2 provides this at 240x121 and 1440x721)
ERA5_1440_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
# Also load climatology for ACC
ERA5_CLIMATOLOGY_URL = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"

# S3 Configuration
S3_BUCKET = "ekasteleyn-aurora-predictions"
S3_FOLDER = "aurora_hres_validation"
S3_ENDPOINT = "https://ceph-gw.science.uva.nl:8000"

RESULTS_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

# Variable mapping: Aurora names -> ERA5 names
VAR_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "q": "specific_humidity",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}

# Key variables to evaluate
EVAL_VARS = [
    ("z", 500),   # Geopotential at 500 hPa
    ("t", 850),   # Temperature at 850 hPa
    ("u", 850),   # U-wind at 850 hPa
    ("v", 850),   # V-wind at 850 hPa
    ("2t", None), # 2m temperature
    ("msl", None), # Mean sea level pressure
]


def get_s3_client():
    """Initialize S3 client with credentials from .env file."""
    load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/scripts/steering/.env")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("UVA_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("UVA_S3_SECRET_KEY"),
    )


def list_aurora_predictions(s3_client):
    """List all Aurora prediction files on S3."""
    nc_files = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FOLDER):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".nc") and "aurora_pred" in key:
                nc_files.append(key)
    return sorted(nc_files)


def parse_filename(filename):
    """Parse Aurora prediction filename.
    
    Format: aurora_pred_YYYYMMDD_HHMM_stepXX_XXXh.nc
    Returns: (init_datetime, step, lead_hours)
    """
    name = Path(filename).stem
    parts = name.split("_")
    # aurora_pred_20221231_0000_step18_108h
    date_str = parts[2]  # YYYYMMDD
    time_str = parts[3]  # HHMM
    step = int(parts[4].replace("step", ""))
    lead_hours = int(parts[5].replace("h", ""))
    
    init_dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
    return init_dt, step, lead_hours


def download_from_s3(s3_client, s3_key, local_path):
    """Download file from S3."""
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))


def compute_lat_weights(lat):
    """Compute latitude weights for weighted averaging (cos(lat))."""
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()  # Normalize
    return xr.DataArray(weights, dims=["latitude"], coords={"latitude": lat})


def compute_rmse(pred, truth, lat):
    """Compute latitude-weighted RMSE."""
    weights = compute_lat_weights(lat)
    diff_sq = (pred - truth) ** 2
    weighted_mse = (diff_sq * weights).mean()
    return float(np.sqrt(weighted_mse))


def compute_acc(pred, truth, climatology, lat):
    """Compute Anomaly Correlation Coefficient (ACC)."""
    weights = compute_lat_weights(lat)
    
    pred_anom = pred - climatology
    truth_anom = truth - climatology
    
    # Weighted covariance
    cov = ((pred_anom * truth_anom) * weights).mean()
    
    # Weighted standard deviations
    std_pred = np.sqrt(((pred_anom ** 2) * weights).mean())
    std_truth = np.sqrt(((truth_anom ** 2) * weights).mean())
    
    if std_pred * std_truth > 0:
        return float(cov / (std_pred * std_truth))
    return np.nan


def evaluate_prediction(aurora_ds, era5_ds, clim_ds, init_time, lead_hours):
    """Evaluate a single Aurora prediction against ERA5."""
    valid_time = pd.Timestamp(init_time) + pd.Timedelta(hours=lead_hours)
    
    # Get ERA5 truth at valid time
    era5_slice = era5_ds.sel(time=valid_time, method="nearest")
    
    # Get climatology for the same day-of-year and hour
    doy = valid_time.dayofyear
    hour = valid_time.hour
    clim_slice = clim_ds.sel(dayofyear=doy, hour=hour)
    
    lat = aurora_ds.latitude.values
    
    results = {}
    
    for var, level in EVAL_VARS:
        if var not in aurora_ds:
            continue
            
        era5_var = VAR_MAP.get(var, var)
        
        try:
            # Get prediction
            if level is not None:
                pred = aurora_ds[var].sel(level=level).values
            else:
                pred = aurora_ds[var].values
            
            # Get truth
            if level is not None:
                truth = era5_slice[era5_var].sel(level=level).values
                clim = clim_slice[era5_var].sel(level=level).values
            else:
                truth = era5_slice[era5_var].values
                clim = clim_slice[era5_var].values
            
            # Ensure shapes match
            if pred.shape != truth.shape:
                continue
            
            # Compute metrics
            key = f"{var}_{level}hPa" if level else var
            results[f"{key}_rmse"] = compute_rmse(pred, truth, lat)
            results[f"{key}_acc"] = compute_acc(pred, truth, clim, lat)
            
        except Exception as e:
            # Skip if variable/level not available
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Aurora 2022 predictions")
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path"
    )
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / "aurora_2022_evaluation.csv"
    
    print("=" * 70)
    print("  AURORA 2022 EVALUATION")
    print("=" * 70)
    
    # Initialize S3 client
    print("\n[1/4] Connecting to S3...")
    s3_client = get_s3_client()
    
    # List prediction files
    print("\n[2/4] Listing prediction files...")
    nc_files = list_aurora_predictions(s3_client)
    print(f"  Found {len(nc_files)} prediction files")
    
    if args.max_files:
        nc_files = nc_files[:args.max_files]
        print(f"  Processing first {len(nc_files)} files (--max-files)")
    
    # Load ERA5 dataset
    print("\n[3/4] Loading ERA5 dataset...")
    era5_ds = xr.open_zarr(fsspec.get_mapper(ERA5_1440_URL), chunks=None)
    print("  ✓ ERA5 loaded")
    
    # Load climatology
    print("  Loading ERA5 climatology...")
    clim_ds = xr.open_zarr(fsspec.get_mapper(ERA5_CLIMATOLOGY_URL), chunks=None)
    print("  ✓ Climatology loaded")
    
    # Process files
    print("\n[4/4] Evaluating predictions...")
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, s3_key in enumerate(nc_files):
            filename = Path(s3_key).name
            local_path = Path(tmpdir) / filename
            
            try:
                # Parse filename
                init_time, step, lead_hours = parse_filename(filename)
                
                if i % 50 == 0 or i == len(nc_files) - 1:
                    print(f"  [{i+1}/{len(nc_files)}] {filename}")
                
                # Download file
                download_from_s3(s3_client, s3_key, local_path)
                
                # Load prediction
                aurora_ds = xr.open_dataset(local_path)
                
                # Evaluate
                metrics = evaluate_prediction(
                    aurora_ds, era5_ds, clim_ds, init_time, lead_hours
                )
                
                # Add metadata
                metrics["date"] = init_time.strftime("%Y-%m-%d")
                metrics["init_hour"] = init_time.hour
                metrics["step"] = step
                metrics["lead_hours"] = lead_hours
                metrics["valid_time"] = (init_time + pd.Timedelta(hours=lead_hours)).strftime("%Y-%m-%d %H:%M")
                
                results.append(metrics)
                
                # Clean up
                aurora_ds.close()
                local_path.unlink()
                
            except Exception as e:
                print(f"  ⚠ Error processing {filename}: {e}")
                continue
    
    # Save results
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\n⚠ No results to save")
        return
    
    # Reorder columns
    meta_cols = ["date", "init_hour", "step", "lead_hours", "valid_time"]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    df = df[[c for c in meta_cols if c in df.columns] + sorted(metric_cols)]
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} evaluations to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY (Mean RMSE by Lead Time)")
    print("=" * 70)
    
    for lead in sorted(df["lead_hours"].unique()):
        sub = df[df["lead_hours"] == lead]
        print(f"\n  Lead +{lead}h (n={len(sub)}):")
        for col in ["z_500hPa_rmse", "t_850hPa_rmse", "2t_rmse"]:
            if col in sub.columns:
                print(f"    {col}: {sub[col].mean():.2f}")


if __name__ == "__main__":
    main()
