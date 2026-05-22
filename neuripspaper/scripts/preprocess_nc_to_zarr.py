#!/usr/bin/env python3
"""
Pre-process individual Aurora 2D/3D flat NetCDF predictions into a single 5D 
Zarr store (time, prediction_timedelta, level, latitude, longitude)
compatible with WeatherBench2 xarray structures.
"""

import os
import re
import pandas as pd
import xarray as xr
import s3fs
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List
from dask.diagnostics import ProgressBar
from dask.distributed import Client

def preprocess_single_file(ds: xr.Dataset) -> xr.Dataset:
    """
    Extracts the 'time' (init time) and 'prediction_timedelta' (lead time)
    from the file attributes and adds them as dimensions to the dataset.
    
    The NC file attribute 'valid_time' is the *valid* time of the prediction
    (set by batch_to_dataset as pred.metadata.time[0], which the Aurora
    decoder advances by lead_time at each step). To get the WeatherBench2-
    compatible *initialization* time we subtract the lead hours.
    """
    # Grab the attributes saved by `batch_to_dataset`
    valid_time_str = ds.attrs.get("valid_time")
    lead_hours = ds.attrs.get("lead_hours")
    
    if valid_time_str is None or lead_hours is None:
        raise ValueError("Missing 'valid_time' or 'lead_hours' attributes in the NetCDF.")
    
    print(f"Pre-processing metadata for valid_time: {valid_time_str}, lead: {lead_hours}h", flush=True)

    valid_time = pd.to_datetime(valid_time_str)
    pred_timedelta = pd.to_timedelta(int(lead_hours), unit="h")
    # WeatherBench2 convention: 'time' = initialization time
    init_time = valid_time - pred_timedelta
    
    # Expand dimensions to include (time, prediction_timedelta)
    ds = ds.expand_dims({
        "time": [init_time],
        "prediction_timedelta": [pred_timedelta]
    })
    
    # Remove attributes to prevent conflicts during concatenation
    ds.attrs.pop("valid_time", None)
    ds.attrs.pop("step", None)
    ds.attrs.pop("lead_hours", None)
    
    return ds

def main():
    load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
    
    # Initialize a Dask local cluster to manage the large task graph efficiently
    client = Client()
    print(f"Dask dashboard available at: {client.dashboard_link}", flush=True)

    # 1. Setup S3 filesystem for direct reading and writing
    print("\nSetting up S3 filesystem...")
    s3 = s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": os.environ.get("AWS_ENDPOINT_URL", "https://ceph-gw.science.uva.nl:8000")}
    )
    
    print("Listing NetCDF files directly from S3...", flush=True)
    all_files = s3.glob("ekasteleyn-aurora-predictions/aurora_hres_validation/aurora_pred_2022*.nc")
    
    print(f"Found {len(all_files)} NetCDF files on S3.", flush=True)
    if not all_files:
        return

    # 2. Setup options for xarray workers
    print(f"Opening {len(all_files)} datasets lazily from S3...", flush=True)
    
    # Open file-like objects for xarray to read directly
    open_files = [s3.open(f) for f in all_files]
    
    # Using parallel=True to parallelize dataset metadata reading
    ds_matched = xr.open_mfdataset(
        open_files,
        engine="h5netcdf",
        preprocess=preprocess_single_file,
        combine="by_coords",  # Automatically uses the expanded time and prediction_timedelta coordinates
        parallel=True,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        chunks={} # Lazily load variables
    )
    
    print("Dataset combined representation:")
    print(ds_matched)
    
    # 3. Write directly to a Zarr store on S3
    s3_store = s3fs.S3Map(root="ekasteleyn-aurora-predictions/aurora_2022.zarr", s3=s3, check=False)
    
    print("Writing chunked dataset directly to S3 Zarr store.")
    
    # Adjust chunks for high performance later
    # (time=1, prediction_timedelta=1 forces 1 file/chunk per time/lead combo)
    chunking = {
        "time": 1, 
        "prediction_timedelta": 1,
        "level": -1,       
        "latitude": -1,
        "longitude": -1
    }
    ds_matched = ds_matched.chunk(chunking)
    
    # Write to S3 Zarr with a progress bar
    print("Beginning computation and writing directly to S3 Zarr...", flush=True)
    with ProgressBar():
        ds_matched.to_zarr(store=s3_store, mode="w", consolidated=True)
    print("Zarr write to S3 complete!")

if __name__ == "__main__":
    main()
