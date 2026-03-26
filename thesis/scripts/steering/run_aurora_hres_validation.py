#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Aurora 0.25° Fine-Tuned Validation with IFS HRES T0 Initialization

This script:
1. Loads IFS HRES T0 data for 3 specific dates in 2022
2. Runs Aurora 0.25° Fine-Tuned model (designed for IFS HRES T0)
3. Saves predictions as NetCDF files
4. Compares predictions against WeatherBench2 Aurora benchmark results

Important: Aurora 0.25° Fine-Tuned should ONLY be used with IFS HRES T0 data
for optimal performance. IFS HRES T0 is NOT the same as IFS HRES analysis.

Required variables:
- Surface: 2t, 10u, 10v, msl
- Static: lsm, slt, z (from ERA5, available on HuggingFace)
- Atmospheric: t, u, v, q, z
- Pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000

WeatherBench2 Aurora contains only levels 500, 700, 850 hPa, so comparison
is limited to those levels.

Usage:
    python run_aurora_hres_validation.py [--dates YYYY-MM-DD ...]
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

# Set memory allocation config before importing Aurora
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from aurora import Aurora, Batch, Metadata, rollout

# ============================================================================
# Configuration
# ============================================================================

# IFS HRES T0 data from WeatherBench2
HRES_T0_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

# WeatherBench2 Aurora benchmark results for comparison
WB2_AURORA_URL = "gs://weatherbench2/datasets/aurora/2022_6h-1440x721.zarr"

# Local cache path
CACHE_PATH = Path.home() / "downloads" / "hres_t0"

# Output directory
OUTPUT_DIR = Path(f"/scratch-shared/{os.environ.get('USER', 'ekasteleyn')}/aurora_hres_validation")

# Default validation dates (3 dates spread across 2022)
DEFAULT_DATES = [
    "2022-01-15",  # Winter
    "2022-05-15",  # Spring
    "2022-09-15",  # Fall
]

# Lead times to evaluate (in hours) - 6h steps
LEAD_TIMES = [6, 12, 18]  # 1, 2, 3 rollout steps

# Pressure levels available in WB2 Aurora for comparison
WB2_LEVELS = [500, 700, 850]

# Variable mappings
SURF_VARS_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}

ATMOS_VARS_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_static_vars(cache_path: Path) -> dict[str, torch.Tensor]:
    """Load static variables from HuggingFace (ERA5-based)."""
    static_file = cache_path / "static.nc"
    
    if not static_file.exists():
        print("  Downloading static variables from HuggingFace...")
        path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
        with open(path, "rb") as f:
            static_vars = pickle.load(f)
        
        # Save as NetCDF for future use
        ds_static = xr.Dataset(
            data_vars={k: (["latitude", "longitude"], v) for k, v in static_vars.items()},
            coords={
                "latitude": ("latitude", np.linspace(90, -90, 721)),
                "longitude": ("longitude", np.linspace(0, 360, 1440, endpoint=False)),
            },
        )
        ds_static.to_netcdf(str(static_file))
        print(f"  ✓ Static variables cached: {static_file}")
    
    ds = xr.open_dataset(static_file, engine="netcdf4")
    return {
        "z": torch.from_numpy(ds["z"].values).float(),
        "slt": torch.from_numpy(ds["slt"].values).float(),
        "lsm": torch.from_numpy(ds["lsm"].values).float(),
    }


def load_hres_t0_batch(
    date_str: str,
    cache_path: Path,
    static_vars: dict[str, torch.Tensor],
) -> Batch:
    """
    Load IFS HRES T0 data for a specific date.
    
    Downloads two consecutive 6-hourly time steps (t-6h and t) as required by Aurora.
    Uses 12:00 UTC as the initialization time.
    """
    cache_path.mkdir(parents=True, exist_ok=True)
    
    surf_file = cache_path / f"{date_str}-surface.nc"
    atmos_file = cache_path / f"{date_str}-atmospheric.nc"
    
    ds = None
    
    # Download surface variables if not cached
    if not surf_file.exists():
        print(f"  Downloading surface variables for {date_str}...")
        ds = xr.open_zarr(fsspec.get_mapper(HRES_T0_URL), chunks=None)
        ds_surf = ds[list(SURF_VARS_MAP.values())].sel(time=date_str).compute()
        ds_surf.to_netcdf(str(surf_file))
        print(f"    ✓ Cached: {surf_file.name}")
    
    # Download atmospheric variables if not cached
    if not atmos_file.exists():
        print(f"  Downloading atmospheric variables for {date_str}...")
        ds = ds or xr.open_zarr(fsspec.get_mapper(HRES_T0_URL), chunks=None)
        ds_atmos = ds[list(ATMOS_VARS_MAP.values())].sel(time=date_str).compute()
        ds_atmos.to_netcdf(str(atmos_file))
        print(f"    ✓ Cached: {atmos_file.name}")
    
    # Load cached data
    surf_ds = xr.open_dataset(surf_file, engine="netcdf4")
    atmos_ds = xr.open_dataset(atmos_file, engine="netcdf4")
    
    # Select time indices: use index 2 (12:00 UTC) and index 1 (06:00 UTC)
    # This gives us the two consecutive 6-hourly time steps Aurora needs
    i = 2  # 12:00 UTC
    
    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare variable: select times, add batch dim, flip lat, convert to tensor."""
        # Select times i-1 and i, add batch dimension, flip latitude (must be decreasing)
        return torch.from_numpy(x[[i - 1, i]][None][..., ::-1, :].copy()).float()
    
    # Get the actual initialization time
    init_time = surf_ds.time.values[i]
    init_time_dt = pd.to_datetime(init_time).to_pydatetime()
    
    return Batch(
        surf_vars={
            k: _prepare(surf_ds[v].values) for k, v in SURF_VARS_MAP.items()
        },
        static_vars=static_vars,
        atmos_vars={
            k: _prepare(atmos_ds[v].values) for k, v in ATMOS_VARS_MAP.items()
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_ds.longitude.values.copy()),
            time=(init_time_dt,),
            atmos_levels=tuple(int(lvl) for lvl in atmos_ds.level.values),
        ),
    )


# ============================================================================
# Prediction
# ============================================================================

def run_prediction(
    model: Aurora,
    batch: Batch,
    num_steps: int,
    device: str,
) -> list[Batch]:
    """Run Aurora rollout for num_steps and return all predictions."""
    batch = batch.to(device)
    
    # Use official rollout generator with inference mode for memory efficiency
    predictions = []
    with torch.inference_mode():
        for pred in rollout(model, batch, num_steps):
            # Move to CPU immediately to free GPU memory
            pred_cpu = pred.to("cpu")
            predictions.append(pred_cpu)
            del pred
            torch.cuda.empty_cache()
    
    return predictions


def batch_to_dataset(batch: Batch, lead_hours: int) -> xr.Dataset:
    """Convert Aurora Batch to xarray Dataset for saving/comparison."""
    lat = batch.metadata.lat.cpu().numpy()
    lon = batch.metadata.lon.cpu().numpy()
    levels = list(batch.metadata.atmos_levels)
    
    data_vars = {}
    
    # Surface variables (take last time step)
    for name, tensor in batch.surf_vars.items():
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0, -1]  # (batch, time, lat, lon) -> (lat, lon)
        elif arr.ndim == 3:
            arr = arr[0]
        data_vars[name] = (["latitude", "longitude"], arr)
    
    # Atmospheric variables (take last time step)
    for name, tensor in batch.atmos_vars.items():
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 5:
            arr = arr[0, -1]  # (batch, time, level, lat, lon) -> (level, lat, lon)
        elif arr.ndim == 4:
            arr = arr[0]
        data_vars[name] = (["level", "latitude", "longitude"], arr)
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "latitude": lat,
            "longitude": lon,
            "level": levels,
        },
    )
    ds.attrs["lead_hours"] = lead_hours
    ds.attrs["model"] = "Aurora 0.25 Fine-Tuned"
    ds.attrs["init_data"] = "IFS HRES T0"
    
    return ds


# ============================================================================
# Comparison with WeatherBench2
# ============================================================================

def compute_rmse(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> float:
    """Compute latitude-weighted RMSE."""
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    
    diff_sq = (pred - truth) ** 2
    
    # Weight by latitude (broadcast for pressure level dimension if present)
    if diff_sq.ndim == 2:  # (lat, lon)
        weighted = diff_sq * weights[:, None]
    elif diff_sq.ndim == 3:  # (level, lat, lon)
        weighted = diff_sq * weights[None, :, None]
    else:
        weighted = diff_sq
    
    return float(np.sqrt(np.nanmean(weighted)))


def compare_with_wb2(
    predictions: dict[str, xr.Dataset],
    init_dates: list[str],
    lead_hours: list[int],
) -> pd.DataFrame:
    """
    Compare our predictions with WeatherBench2 Aurora benchmark.
    
    Note: WB2 Aurora only contains levels 500, 700, 850 hPa.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON WITH WEATHERBENCH2 AURORA BENCHMARK")
    print("=" * 70)
    
    results = []
    
    try:
        print("  Opening WB2 Aurora dataset...")
        wb2 = xr.open_zarr(fsspec.get_mapper(WB2_AURORA_URL), chunks=None)
        print(f"  ✓ WB2 Aurora dataset opened")
        print(f"    Variables: {list(wb2.data_vars)[:5]}...")
        print(f"    Levels: {list(wb2.level.values)}")
    except Exception as e:
        print(f"  ⚠ Could not open WB2 Aurora: {e}")
        print("  Skipping comparison.")
        return pd.DataFrame()
    
    for date_str in init_dates:
        for lead_h in lead_hours:
            key = f"{date_str}_lead{lead_h:02d}h"
            
            if key not in predictions:
                continue
            
            pred_ds = predictions[key]
            init_time = pd.to_datetime(f"{date_str}T12:00:00")
            valid_time = init_time + timedelta(hours=lead_h)
            
            try:
                # Get WB2 Aurora prediction for this valid time
                wb2_slice = wb2.sel(time=valid_time, method="nearest").compute()
                
                lat = pred_ds.latitude.values
                
                # Compare geopotential at levels 500, 700, 850
                for level in WB2_LEVELS:
                    if level not in pred_ds.level.values:
                        continue
                    
                    pred_z = pred_ds["z"].sel(level=level).values
                    
                    # WB2 uses "geopotential" name
                    if "geopotential" in wb2_slice:
                        wb2_z = wb2_slice["geopotential"].sel(level=level).values
                    elif "z" in wb2_slice:
                        wb2_z = wb2_slice["z"].sel(level=level).values
                    else:
                        continue
                    
                    # Ensure same grid orientation
                    if wb2_z.shape != pred_z.shape:
                        print(f"    ⚠ Shape mismatch at {key} level {level}: "
                              f"pred={pred_z.shape}, wb2={wb2_z.shape}")
                        continue
                    
                    diff = compute_rmse(pred_z, wb2_z, lat)
                    
                    results.append({
                        "date": date_str,
                        "lead_hours": lead_h,
                        "level": level,
                        "variable": "z",
                        "rmse_vs_wb2": diff,
                    })
                
                # Compare temperature at levels 500, 700, 850
                for level in WB2_LEVELS:
                    if level not in pred_ds.level.values:
                        continue
                    
                    pred_t = pred_ds["t"].sel(level=level).values
                    
                    if "temperature" in wb2_slice:
                        wb2_t = wb2_slice["temperature"].sel(level=level).values
                    elif "t" in wb2_slice:
                        wb2_t = wb2_slice["t"].sel(level=level).values
                    else:
                        continue
                    
                    if wb2_t.shape != pred_t.shape:
                        continue
                    
                    diff = compute_rmse(pred_t, wb2_t, lat)
                    
                    results.append({
                        "date": date_str,
                        "lead_hours": lead_h,
                        "level": level,
                        "variable": "t",
                        "rmse_vs_wb2": diff,
                    })
                
                print(f"    ✓ {key}: compared at levels {WB2_LEVELS}")
                
            except Exception as e:
                print(f"    ⚠ {key}: {e}")
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print("\n  SUMMARY (RMSE of our prediction vs WB2 Aurora):")
        print("  " + "-" * 60)
        summary = df.groupby(["variable", "level"])["rmse_vs_wb2"].agg(["mean", "std"])
        print(summary.to_string())
        print("\n  Note: Small RMSE indicates our run matches WB2 Aurora benchmark.")
        print("  Values should be very close to 0 if using identical inputs.")
    
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Aurora 0.25° Fine-Tuned with IFS HRES T0 initialization"
    )
    parser.add_argument(
        "--dates", nargs="+", default=DEFAULT_DATES,
        help=f"Dates to process (YYYY-MM-DD). Default: {DEFAULT_DATES}"
    )
    parser.add_argument(
        "--num-steps", type=int, default=3,
        help="Number of 6h rollout steps (default: 3 = 18h forecast)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache directory for downloads (default: {CACHE_PATH})"
    )
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip comparison with WB2 Aurora benchmark"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu). Default: auto-detect"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    cache_path = Path(args.cache_dir) if args.cache_dir else CACHE_PATH
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  AURORA 0.25° FINE-TUNED - IFS HRES T0 VALIDATION")
    print("=" * 70)
    print(f"  Dates: {args.dates}")
    print(f"  Rollout steps: {args.num_steps} (= {args.num_steps * 6}h forecast)")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Cache: {cache_path}")
    print("=" * 70)
    
    # Load static variables
    print("\n[1/4] Loading static variables...")
    static_vars = load_static_vars(cache_path)
    print(f"  ✓ Static vars loaded: {list(static_vars.keys())}")
    
    # Load model with autocast enabled to reduce memory usage
    print("\n[2/4] Loading Aurora 0.25° Fine-Tuned model...")
    model = Aurora(autocast=True)  # Enable autocast for memory efficiency
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)
    print("  ✓ Model loaded and ready (autocast=True)")
    
    # Process each date
    all_predictions = {}
    
    print(f"\n[3/4] Running predictions for {len(args.dates)} dates...")
    
    for idx, date_str in enumerate(args.dates, 1):
        print(f"\n  [{idx}/{len(args.dates)}] {date_str}")
        
        try:
            # Load batch
            batch = load_hres_t0_batch(date_str, cache_path, static_vars)
            init_time = batch.metadata.time[0]
            print(f"    Init time: {init_time}")
            print(f"    Levels: {batch.metadata.atmos_levels}")
            
            # Run predictions
            predictions = run_prediction(model, batch, args.num_steps, device)
            
            # Save predictions
            for step, pred in enumerate(predictions, 1):
                lead_h = step * 6
                key = f"{date_str}_lead{lead_h:02d}h"
                
                pred_ds = batch_to_dataset(pred, lead_h)
                pred_ds.attrs["init_time"] = str(init_time)
                pred_ds.attrs["valid_time"] = str(init_time + timedelta(hours=lead_h))
                
                # Save to NetCDF
                out_file = output_dir / f"aurora_hres_{date_str.replace('-', '')}_step{step:02d}.nc"
                pred_ds.to_netcdf(out_file)
                print(f"    ✓ Step {step} (+{lead_h}h): {out_file.name}")
                
                all_predictions[key] = pred_ds
            
            # Clean up GPU memory
            del batch, predictions
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare with WB2
    if not args.skip_comparison and len(all_predictions) > 0:
        print("\n[4/4] Comparing with WeatherBench2 Aurora benchmark...")
        lead_hours = [6 * (i + 1) for i in range(args.num_steps)]
        df_comparison = compare_with_wb2(all_predictions, args.dates, lead_hours)
        
        if len(df_comparison) > 0:
            csv_path = output_dir / "wb2_comparison.csv"
            df_comparison.to_csv(csv_path, index=False)
            print(f"\n  ✓ Comparison saved: {csv_path}")
    else:
        print("\n[4/4] Skipping WB2 comparison")
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Predictions saved to: {output_dir}")
    print(f"  Files: {len(list(output_dir.glob('aurora_hres_*.nc')))} NetCDF files")


if __name__ == "__main__":
    main()
