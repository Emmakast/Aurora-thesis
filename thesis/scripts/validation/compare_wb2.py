#!/usr/bin/env python3
"""Compare Aurora predictions against WeatherBench2 reference data."""

import os
import tempfile
from pathlib import Path

import boto3
import fsspec
import numpy as np
import xarray as xr
from dotenv import load_dotenv

# WB2 Aurora reference
WB2_AURORA_URL = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"

# Variable mapping: our names -> WB2 names
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

# Pressure levels available in WB2
WB2_LEVELS = [500, 700, 850]


def download_from_s3(s3_client, bucket, key, local_path):
    """Download file from S3."""
    s3_client.download_file(bucket, key, str(local_path))


# Global WB2 dataset cache
_wb2_ds = None

def get_wb2_ds():
    """Get cached WB2 dataset."""
    global _wb2_ds
    if _wb2_ds is None:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        print("  Loading WB2 Aurora dataset (one-time)...")
        _wb2_ds = xr.open_zarr(fsspec.get_mapper(WB2_AURORA_URL), chunks=None)
        print("  ✓ WB2 dataset loaded")
    return _wb2_ds


def load_wb2_prediction(init_time, lead_hours, variable, level=None):
    """Load WB2 Aurora reference prediction."""
    ds = get_wb2_ds()
    
    # WB2 uses "time" for init time and "prediction_timedelta" for lead
    init_str = init_time.strftime("%Y-%m-%d %H:%M")
    lead_td = np.timedelta64(lead_hours, 'h')
    
    wb2_var = VAR_MAP.get(variable, variable)
    
    if level is not None:
        data = ds[wb2_var].sel(time=init_str, prediction_timedelta=lead_td, level=level)
    else:
        data = ds[wb2_var].sel(time=init_str, prediction_timedelta=lead_td)
    
    return data.values


def compare_predictions(our_path, init_time, lead_hours):
    """Compare our prediction file against WB2 reference."""
    our_ds = xr.open_dataset(our_path)
    
    results = {}
    
    # Compare atmospheric variables at WB2 levels
    for our_var in ["t", "u", "v", "z", "q"]:
        if our_var not in our_ds:
            continue
        
        for level in WB2_LEVELS:
            if level not in our_ds.level.values:
                continue
            
            our_data = our_ds[our_var].sel(level=level).values
            
            try:
                wb2_data = load_wb2_prediction(init_time, lead_hours, our_var, level)
                
                # Compute error metrics
                diff = our_data - wb2_data
                mae = np.abs(diff).mean()
                rmse = np.sqrt((diff**2).mean())
                max_err = np.abs(diff).max()
                
                results[f"{our_var}_{level}hPa"] = {
                    "MAE": float(mae),
                    "RMSE": float(rmse),
                    "Max Error": float(max_err),
                }
            except Exception as e:
                results[f"{our_var}_{level}hPa"] = {"error": str(e)}
    
    # Compare surface variables
    for our_var in ["2t", "10u", "10v", "msl"]:
        if our_var not in our_ds:
            continue
        
        our_data = our_ds[our_var].values
        
        try:
            wb2_data = load_wb2_prediction(init_time, lead_hours, our_var)
            
            diff = our_data - wb2_data
            mae = np.abs(diff).mean()
            rmse = np.sqrt((diff**2).mean())
            max_err = np.abs(diff).max()
            
            results[our_var] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "Max Error": float(max_err),
            }
        except Exception as e:
            results[our_var] = {"error": str(e)}
    
    our_ds.close()
    return results


def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="ekasteleyn-aurora-predictions")
    parser.add_argument("--folder", default="aurora_hres_validation")
    parser.add_argument("--local-file", help="Use local .nc file instead of S3")
    args = parser.parse_args()
    
    load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/scripts/steering/.env")
    
    # Setup S3 client
    s3_client = boto3.client('s3',
        endpoint_url='https://ceph-gw.science.uva.nl:8000',
        aws_access_key_id=os.getenv('UVA_S3_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('UVA_S3_SECRET_KEY')
    )
    
    # List available prediction files
    print("Listing prediction files on S3...")
    response = s3_client.list_objects_v2(Bucket=args.bucket, Prefix=args.folder)
    
    nc_files = [obj['Key'] for obj in response.get('Contents', []) 
                if obj['Key'].endswith('.nc')]
    
    print(f"Found {len(nc_files)} prediction files")
    
    # Pick just one file to compare (12:00 init, step 1 for quick test)
    test_files = [f for f in nc_files if "1200_step01" in f][:1]
    if not test_files:
        test_files = nc_files[:1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for s3_key in test_files:
            filename = Path(s3_key).name
            local_path = Path(tmpdir) / filename
            
            # Parse filename: aurora_pred_YYYYMMDD_HHMM_stepXX_XXXh.nc
            parts = filename.replace(".nc", "").split("_")
            date_str = parts[2]  # YYYYMMDD
            time_str = parts[3]  # HHMM
            lead_hours = int(parts[5].replace("h", ""))  # XXXh
            
            init_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            
            print(f"\n{'='*60}")
            print(f"Comparing: {filename}")
            print(f"Init: {init_time}, Lead: +{lead_hours}h")
            print(f"{'='*60}")
            
            # Download file
            print("  Downloading from S3...")
            download_from_s3(s3_client, args.bucket, s3_key, local_path)
            
            # Compare
            print("  Comparing against WB2...")
            results = compare_predictions(local_path, init_time, lead_hours)
            
            # Print results
            all_zero = True
            for var, metrics in results.items():
                if "error" in metrics:
                    print(f"  {var}: ERROR - {metrics['error']}")
                    all_zero = False
                else:
                    mae = metrics["MAE"]
                    rmse = metrics["RMSE"]
                    max_err = metrics["Max Error"]
                    status = "✓" if mae < 1e-5 else "⚠"
                    print(f"  {status} {var}: MAE={mae:.2e}, RMSE={rmse:.2e}, Max={max_err:.2e}")
                    if mae > 1e-5:
                        all_zero = False
            
            if all_zero:
                print("\n  ✓ PERFECT MATCH - predictions identical to WB2")
            else:
                print("\n  ⚠ DIFFERENCES DETECTED")


if __name__ == "__main__":
    main()
