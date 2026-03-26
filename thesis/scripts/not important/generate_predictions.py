#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Aurora Weather Prediction Generator

Generates Aurora weather predictions for specified dates using ERA5 data.
Outputs predictions as NetCDF files for downstream analysis (e.g., GDAM conservation checks).

Configuration:
- Uses ERA5 Zarr data from the managed datasets
- Uses Aurora 0.25° model (AuroraSmall for memory efficiency)
- Performs multi-step rollouts with 6-hour intervals

Author: Aurora Prediction Generation Script
Date: 2025-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Mitigate CUDA memory fragmentation
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================================
# Configuration
# ============================================================================

# Data paths
ZARR_PATH = "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
STATIC_PATH = Path.home() / "downloads" / "era5" / "static.nc"

# Output directory
OUTPUT_DIR = Path.home() / "aurora_thesis" / "predictions"

# Target dates: First day of each month, Jan-Oct 2020
TARGET_DATES = [
    "2020-01-01",
    "2020-02-01",
    "2020-03-01",
    "2020-04-01",
    "2020-05-01",
    "2020-06-01",
    "2020-07-01",
    "2020-08-01",
    "2020-09-01",
    "2020-10-01",
]

# Rollout configuration
NUM_ROLLOUT_STEPS = 3  # Number of prediction steps
STEP_HOURS = 6  # Hours between each step

# ERA5 variable name mapping: Aurora key → ERA5 Zarr variable name
# Based on WeatherBench / Aurora naming conventions
VAR_MAP_ERA5_TO_AURORA = {
    # Surface variables
    "2t": "2m_temperature",           # 2-meter temperature
    "10u": "10m_u_component_of_wind",  # 10-meter U wind
    "10v": "10m_v_component_of_wind",  # 10-meter V wind
    "msl": "mean_sea_level_pressure",  # Mean sea level pressure (synoptic analysis)
    "sp": "surface_pressure",          # Surface pressure (for mass conservation)
    # Atmospheric (pressure level) variables
    "t": "temperature",                # Temperature at pressure levels
    "u": "u_component_of_wind",        # U component of wind
    "v": "v_component_of_wind",        # V component of wind
    "q": "specific_humidity",          # Specific humidity
    "z": "geopotential",               # Geopotential
}


# ============================================================================
# Aurora Batch Creation
# ============================================================================

def get_aurora_batch(
    date_str: str,
    zarr_path: str = ZARR_PATH,
    static_path: Path = STATIC_PATH,
    init_hour: int = 6
):
    """
    Create an Aurora Batch object for a specific date.
    
    Aurora requires two consecutive time steps as input (for temporal context).
    This function selects t-6h and t as the input times.
    
    Parameters
    ----------
    date_str : str
        Target date in YYYY-MM-DD format
    zarr_path : str
        Path to ERA5 Zarr dataset
    static_path : Path
        Path to static variables NetCDF file
    init_hour : int
        Initialization hour (default 6 = 06:00 UTC)
        
    Returns
    -------
    aurora.Batch
        Batch object ready for model inference
    """
    # Import Aurora here to avoid issues if not installed
    from aurora import AuroraSmall, Batch, Metadata
    
    print(f"\n  Loading data for {date_str}...")
    
    # Open ERA5 Zarr dataset
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    # Define target time and the preceding time step
    target_time = pd.to_datetime(f"{date_str}T{init_hour:02d}:00:00")
    prev_time = target_time - timedelta(hours=STEP_HOURS)
    request_times = [prev_time, target_time]
    
    print(f"    Requesting times: {prev_time} → {target_time}")
    
    # Select the time steps
    try:
        frame = ds.sel(time=request_times, method="nearest").load()
        frame = frame.sortby("time")
        
        if frame.time.size < 2:
            raise ValueError(f"Only found {frame.time.size} time steps; need 2")
        
        # Verify we got the expected times
        actual_times = pd.to_datetime(frame.time.values)
        print(f"    Loaded times: {actual_times[0]} → {actual_times[1]}")
        
    except Exception as e:
        raise ValueError(f"Failed to load {date_str}: {e}")
    
    # Load static variables
    static = xr.open_dataset(static_path, engine="netcdf4")
    
    # Log shapes for debugging
    print(f"    Grid: {len(frame.latitude)} × {len(frame.longitude)}")
    print(f"    Pressure levels: {frame.level.values.tolist()}")
    
    # Build surface variables tensor dict
    # Shape: (batch=1, time=2, lat, lon)
    surf_vars = {}
    for aurora_key in ["2t", "10u", "10v", "msl", "sp"]:
        era5_name = VAR_MAP_ERA5_TO_AURORA[aurora_key]
        data = frame[era5_name].values  # (time, lat, lon)
        surf_vars[aurora_key] = torch.from_numpy(data).unsqueeze(0).float()
    
    # Build atmospheric variables tensor dict
    # Shape: (batch=1, time=2, level, lat, lon)
    atmos_vars = {}
    for aurora_key in ["t", "u", "v", "q", "z"]:
        era5_name = VAR_MAP_ERA5_TO_AURORA[aurora_key]
        data = frame[era5_name].values  # (time, level, lat, lon)
        atmos_vars[aurora_key] = torch.from_numpy(data).unsqueeze(0).float()
    
    # Build static variables tensor dict
    # Shape: (lat, lon) - no batch or time dimension
    static_vars = {
        "z": torch.from_numpy(static["z"].values[0]).float(),
        "slt": torch.from_numpy(static["slt"].values[0]).float(),
        "lsm": torch.from_numpy(static["lsm"].values[0]).float(),
    }
    
    # Create metadata
    metadata = Metadata(
        lat=torch.from_numpy(frame.latitude.values),
        lon=torch.from_numpy(frame.longitude.values),
        time=tuple(pd.to_datetime(frame.time.values).to_pydatetime()),
        atmos_levels=tuple(int(lvl) for lvl in frame.level.values),
    )
    
    # Create and return batch
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )
    
    return batch


# ============================================================================
# Prediction to xarray Dataset
# ============================================================================

def batch_to_dataset(
    batch,
    valid_time: datetime,
    init_time: datetime,
    step: int
) -> xr.Dataset:
    """
    Convert Aurora Batch prediction output to xarray Dataset.
    
    Note: Aurora crops the input grid during inference (e.g., 721→720 lat points),
    so we extract coordinates directly from the prediction batch's metadata.
    
    Parameters
    ----------
    batch : aurora.Batch
        Prediction batch from model.forward()
    valid_time : datetime
        Valid time of the prediction
    init_time : datetime
        Initialization time
    step : int
        Rollout step number
        
    Returns
    -------
    xr.Dataset
        Dataset with prediction variables
    """
    # Extract coordinates from the prediction batch's metadata
    # This gives us the actual (cropped) grid dimensions
    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()
    levels = list(batch.metadata.atmos_levels)
    
    coords = {
        "latitude": lat,
        "longitude": lon,
        "level": levels,
        "time": [valid_time],
    }
    
    data_vars = {}
    
    # Surface variables: extract last time step
    for name, tensor in batch.surf_vars.items():
        arr = tensor.detach().cpu().numpy()
        # Shape: (batch, time, lat, lon) → take last time, squeeze batch
        if arr.ndim == 4:
            arr = arr[0, -1]  # (lat, lon)
        elif arr.ndim == 3:
            arr = arr[0]
        
        # Add time dimension for consistency
        arr = arr[np.newaxis, ...]  # (1, lat, lon)
        data_vars[name] = (["time", "latitude", "longitude"], arr)
    
    # Atmospheric variables: extract last time step
    for name, tensor in batch.atmos_vars.items():
        arr = tensor.detach().cpu().numpy()
        # Shape: (batch, time, level, lat, lon) → take last time, squeeze batch
        if arr.ndim == 5:
            arr = arr[0, -1]  # (level, lat, lon)
        elif arr.ndim == 4:
            arr = arr[0]
        
        # Add time dimension
        arr = arr[np.newaxis, ...]  # (1, level, lat, lon)
        data_vars[name] = (["time", "level", "latitude", "longitude"], arr)
    
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add metadata as attributes
    ds.attrs["init_time"] = init_time.isoformat()
    ds.attrs["valid_time"] = valid_time.isoformat()
    ds.attrs["step"] = step
    ds.attrs["lead_hours"] = step * STEP_HOURS
    ds.attrs["model"] = "Aurora 0.25deg"
    ds.attrs["source"] = "Microsoft Aurora Weather Model"
    ds.attrs["grid_note"] = f"Grid cropped from 721 to {len(lat)} lat points by Aurora patch processing"
    
    return ds


# ============================================================================
# Main Prediction Loop
# ============================================================================

def run_predictions(
    dates: list[str],
    num_steps: int = NUM_ROLLOUT_STEPS,
    output_dir: Path = OUTPUT_DIR,
    device: Optional[str] = None,
    save_all_steps: bool = True
) -> list[Path]:
    """
    Generate Aurora predictions for multiple dates.
    
    Parameters
    ----------
    dates : list[str]
        List of dates in YYYY-MM-DD format
    num_steps : int
        Number of rollout steps to perform
    output_dir : Path
        Directory to save predictions
    device : str, optional
        Device to run on (cuda/cpu)
    save_all_steps : bool
        If True, save each rollout step; if False, save only final step
        
    Returns
    -------
    list[Path]
        List of saved file paths
    """
    from aurora import AuroraSmall
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  AURORA PREDICTION GENERATION")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Dates: {len(dates)}")
    print(f"  Steps per date: {num_steps}")
    print(f"  Step interval: {STEP_HOURS}h")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (do this once, not per date)
    print("\n  Loading Aurora model...")
    model = AuroraSmall()
    model.load_checkpoint()
    model.eval()
    model = model.to(device)
    print("  ✓ Model loaded")
    
    saved_files = []
    
    for date_idx, date_str in enumerate(dates):
        print(f"\n[{date_idx + 1}/{len(dates)}] Processing {date_str}")
        
        try:
            # Load input batch
            batch = get_aurora_batch(date_str)
            init_time = batch.metadata.time[-1]  # Last input time is init time
            
            # Prepare batch for model
            batch = batch.to(device)
            
            # Rollout loop
            current_batch = batch
            for step in range(1, num_steps + 1):
                valid_time = init_time + timedelta(hours=step * STEP_HOURS)
                print(f"    Step {step}/{num_steps}: +{step * STEP_HOURS}h → {valid_time}")
                
                # Apply model transforms and run inference
                with torch.inference_mode():
                    # Apply the batch transform hook (required by Aurora)
                    transformed = model.batch_transform_hook(current_batch)
                    transformed = transformed.type(next(model.parameters()).dtype)
                    transformed = transformed.crop(model.patch_size)
                    
                    # Forward pass
                    prediction = model.forward(transformed)
                
                # Save prediction
                if save_all_steps or step == num_steps:
                    ds = batch_to_dataset(
                        prediction,
                        valid_time=valid_time,
                        init_time=init_time,
                        step=step
                    )
                    
                    # Filename: pred_YYYYMMDD_stepNN.nc
                    date_clean = date_str.replace("-", "")
                    filename = f"pred_{date_clean}_step{step:02d}.nc"
                    filepath = output_dir / filename
                    
                    ds.to_netcdf(filepath)
                    saved_files.append(filepath)
                    print(f"      ✓ Saved: {filepath.name}")
                
                # Use prediction as input for next step (autoregressive rollout)
                current_batch = prediction
            
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {len(saved_files)} files saved")
    print(f"{'='*60}")
    
    return saved_files


# ============================================================================
# Generate ERA5 Ground Truth Files for Comparison
# ============================================================================

def save_era5_ground_truth(
    dates: list[str],
    num_steps: int = NUM_ROLLOUT_STEPS,
    output_dir: Optional[Path] = None,
    zarr_path: str = ZARR_PATH
) -> list[Path]:
    """
    Save ERA5 ground truth files matching the prediction valid times.
    
    This is useful for GDAM conservation comparison.
    
    Parameters
    ----------
    dates : list[str]
        List of initialization dates
    num_steps : int
        Number of steps (to get valid times)
    output_dir : Path, optional
        Output directory (default: OUTPUT_DIR/era5_truth)
    zarr_path : str
        Path to ERA5 Zarr dataset
        
    Returns
    -------
    list[Path]
        List of saved ERA5 files
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "era5_truth"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  SAVING ERA5 GROUND TRUTH")
    print(f"{'='*60}")
    
    ds = xr.open_zarr(zarr_path, consolidated=True)
    saved_files = []
    
    for date_str in dates:
        init_time = pd.to_datetime(f"{date_str}T06:00:00")
        
        for step in range(1, num_steps + 1):
            valid_time = init_time + timedelta(hours=step * STEP_HOURS)
            
            try:
                # Select the valid time
                era5 = ds.sel(time=valid_time, method="nearest").load()
                
                # Convert to dataset with standard variable names
                era5_ds = era5.to_dataset() if hasattr(era5, 'to_dataset') else era5
                
                # Add metadata
                era5_ds.attrs["valid_time"] = str(valid_time)
                era5_ds.attrs["source"] = "ERA5"
                
                # Save
                date_clean = date_str.replace("-", "")
                valid_str = valid_time.strftime("%Y%m%d_%H%M")
                filename = f"era5_{valid_str}.nc"
                filepath = output_dir / filename
                
                era5_ds.to_netcdf(filepath)
                saved_files.append(filepath)
                print(f"  ✓ {filename}")
                
            except Exception as e:
                print(f"  ⚠ Error for {valid_time}: {e}")
    
    return saved_files


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Aurora weather predictions for specified dates"
    )
    parser.add_argument(
        "--dates", type=str, nargs="+", default=None,
        help="List of dates (YYYY-MM-DD). Default: first 10 months of 2020"
    )
    parser.add_argument(
        "--num-steps", type=int, default=NUM_ROLLOUT_STEPS,
        help=f"Number of rollout steps (default: {NUM_ROLLOUT_STEPS})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--save-era5", action="store_true",
        help="Also save ERA5 ground truth for comparison"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Just list what would be generated, don't run"
    )
    args = parser.parse_args()
    
    # Set dates
    dates = args.dates if args.dates else TARGET_DATES
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    if args.list_only:
        print("\nWould generate predictions for:")
        for d in dates:
            print(f"  - {d}")
        print(f"\nOutput directory: {output_dir}")
        print(f"Steps per date: {args.num_steps}")
        print(f"Total files: {len(dates) * args.num_steps}")
        return
    
    # Run predictions
    run_predictions(
        dates=dates,
        num_steps=args.num_steps,
        output_dir=output_dir,
        device=args.device
    )
    
    # Optionally save ERA5 ground truth
    if args.save_era5:
        save_era5_ground_truth(
            dates=dates,
            num_steps=args.num_steps,
            output_dir=output_dir / "era5_truth"
        )


if __name__ == "__main__":
    main()
