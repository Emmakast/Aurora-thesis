#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Generate Aurora predictions for comparison with WeatherBench2 reference.

This script runs Aurora rollouts and saves ONLY predictions (no latents/attention)
for later comparison against the WB2 Aurora reference dataset.

Usage:
    python run_wb2_predictions.py --date 2022-01-15 --num-steps 40
"""

from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import torch
import xarray as xr

from aurora import Aurora, Batch, Metadata, rollout

# Enable TF32 for faster float32 matmul on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DOWNLOAD_PATH = Path.home() / "downloads" / "hres_t0"

# Use SLURM's $TMPDIR or fallback to scratch-shared
base_tmp = os.environ.get('TMPDIR', f"/scratch-shared/{os.environ.get('USER', 'ekasteleyn')}")
OUTPUT_DIR = Path(base_tmp) / "aurora_wb2_predictions"

# WeatherBench2 HRES T0 data (input data source)
WB2_HRES_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading (from official Microsoft example)
# ══════════════════════════════════════════════════════════════════════════════

def download_data(day: str, download_path: Path):
    """Download HRES T0 data for a specific day."""
    download_path.mkdir(parents=True, exist_ok=True)
    
    ds = xr.open_zarr(fsspec.get_mapper(WB2_HRES_URL), chunks=None)
    
    # Download surface-level variables
    if not (download_path / f"{day}-surface-level.nc").exists():
        print(f"    Downloading surface variables...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(download_path / f"{day}-surface-level.nc"))
    
    # Download atmospheric variables
    if not (download_path / f"{day}-atmospheric.nc").exists():
        print(f"    Downloading atmospheric variables...")
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(download_path / f"{day}-atmospheric.nc"))


def download_static(download_path: Path):
    """Download static variables safely via CDS API (Microsoft's exact method)."""
    import cdsapi
    
    if not (download_path / "static.nc").exists():
        print("  Downloading static variables from Copernicus CDS (This only happens ONCE)...")
        try:
            c = cdsapi.Client()
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ["geopotential", "land_sea_mask", "soil_type"],
                    "year": "2023",
                    "month": "01",
                    "day": "01",
                    "time": "00:00",
                    "format": "netcdf",
                },
                str(download_path / "static.nc"),
            )
            print("    ✓ Static variables cached securely")
        except Exception as e:
            print(f"\n  [ERROR] CDS API Failed: {e}")
            print("  Please ensure your ~/.cdsapirc file is configured correctly.")
            raise e


def prepare_batch(day: str, download_path: Path, init_hour: int = 12) -> Batch:
    """Prepare batch following official Microsoft example.
    
    Args:
        day: Date string (YYYY-MM-DD)
        download_path: Path to downloaded data
        init_hour: Initialization hour (0 or 12 UTC)
    """
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")
    
    if init_hour == 0:
        # Init 00:00 UTC: need previous day's 18:00 and current day's 00:00
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prev_surf_ds = xr.open_dataset(download_path / f"{prev_day}-surface-level.nc", engine="netcdf4")
        prev_atmos_ds = xr.open_dataset(download_path / f"{prev_day}-atmospheric.nc", engine="netcdf4")
        
        def _prepare_init00(x_prev: np.ndarray, x_curr: np.ndarray) -> torch.Tensor:
            """Prepare for init=00:00: prev day 18:00 (idx 3) + current day 00:00 (idx 0)."""
            combined = np.stack([x_prev[3], x_curr[0]], axis=0)
            return torch.from_numpy(combined[None][..., ::-1, :].copy())
        
        batch = Batch(
            surf_vars={
                "2t": _prepare_init00(prev_surf_ds["2m_temperature"].values, surf_vars_ds["2m_temperature"].values),
                "10u": _prepare_init00(prev_surf_ds["10m_u_component_of_wind"].values, surf_vars_ds["10m_u_component_of_wind"].values),
                "10v": _prepare_init00(prev_surf_ds["10m_v_component_of_wind"].values, surf_vars_ds["10m_v_component_of_wind"].values),
                "msl": _prepare_init00(prev_surf_ds["mean_sea_level_pressure"].values, surf_vars_ds["mean_sea_level_pressure"].values),
            },
            static_vars={
                "z": torch.from_numpy(static_vars_ds["z"].values[::-1, :].copy()),
                "slt": torch.from_numpy(static_vars_ds["slt"].values[::-1, :].copy()),
                "lsm": torch.from_numpy(static_vars_ds["lsm"].values[::-1, :].copy()),
            },
            atmos_vars={
                "t": _prepare_init00(prev_atmos_ds["temperature"].values, atmos_vars_ds["temperature"].values),
                "u": _prepare_init00(prev_atmos_ds["u_component_of_wind"].values, atmos_vars_ds["u_component_of_wind"].values),
                "v": _prepare_init00(prev_atmos_ds["v_component_of_wind"].values, atmos_vars_ds["v_component_of_wind"].values),
                "q": _prepare_init00(prev_atmos_ds["specific_humidity"].values, atmos_vars_ds["specific_humidity"].values),
                "z": _prepare_init00(prev_atmos_ds["geopotential"].values, atmos_vars_ds["geopotential"].values),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
                lon=torch.from_numpy(surf_vars_ds.longitude.values),
                time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[0],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
            ),
        )
        prev_surf_ds.close()
        prev_atmos_ds.close()
        return batch
    elif init_hour == 12:
        # Init 12:00 UTC: use indices 1 (06:00) and 2 (12:00)
        time_indices = [1, 2]
        init_time_idx = 2
    else:
        raise ValueError(f"init_hour must be 0 or 12, got {init_hour}")
    
    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare a variable with specified time indices."""
        return torch.from_numpy(x[time_indices][None][..., ::-1, :].copy())
    
    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[::-1, :].copy()),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[::-1, :].copy()),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[::-1, :].copy()),
        },
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[init_time_idx],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )
    
    return batch


# ══════════════════════════════════════════════════════════════════════════════
# Prediction Saving
# ══════════════════════════════════════════════════════════════════════════════

def batch_to_dataset(pred: Batch, step: int) -> xr.Dataset:
    """Convert Aurora Batch prediction to xarray Dataset (float32)."""
    lat = pred.metadata.lat.numpy()
    lon = pred.metadata.lon.numpy()
    levels = list(pred.metadata.atmos_levels)
    
    data_vars = {}
    
    # Surface variables
    for name, tensor in pred.surf_vars.items():
        arr = tensor.numpy().astype(np.float32)
        arr = arr[0, 0]  # (batch=1, time=1, lat, lon) -> (lat, lon)
        data_vars[name] = (["latitude", "longitude"], arr)
    
    # Atmospheric variables
    for name, tensor in pred.atmos_vars.items():
        arr = tensor.numpy().astype(np.float32)
        arr = arr[0, 0]  # (batch=1, time=1, level, lat, lon) -> (level, lat, lon)
        data_vars[name] = (["level", "latitude", "longitude"], arr)
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "latitude": lat.astype(np.float32),
            "longitude": lon.astype(np.float32),
            "level": levels,
        },
    )
    ds.attrs["valid_time"] = str(pred.metadata.time[0])
    ds.attrs["step"] = step
    ds.attrs["lead_hours"] = step * 6
    ds.attrs["model"] = "Aurora 0.25 Fine-Tuned"
    
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Aurora predictions for WB2 comparison"
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="Date to process (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--num-steps", type=int, default=40,
        help="Number of rollout steps (default: 40 = 240 hours)"
    )
    parser.add_argument(
        "--init-hours", nargs="+", type=int, default=[0, 12],
        help="Init hours to process (default: [0, 12])"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache directory for downloads (default: {DOWNLOAD_PATH})"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    download_path = Path(args.cache_dir) if args.cache_dir else DOWNLOAD_PATH
    
    output_dir.mkdir(parents=True, exist_ok=True)
    download_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("  AURORA WB2 PREDICTION COMPARISON RUN")
    print("=" * 70)
    print(f"  Date: {args.date}")
    print(f"  Init hours: {args.init_hours}")
    print(f"  Rollout steps: {args.num_steps} ({args.num_steps * 6} hours)")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Cache: {download_path}")
    print("=" * 70)
    
    # Download static variables
    print("\n[1/3] Downloading static variables...")
    download_static(download_path)
    
    # Download data for the date (and previous day if init_hour=0)
    print("\n[2/3] Downloading HRES T0 data...")
    dates_to_download = {args.date}
    if 0 in args.init_hours:
        prev_day = (pd.to_datetime(args.date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        dates_to_download.add(prev_day)
    for day in sorted(dates_to_download):
        print(f"  {day}")
        download_data(day, download_path)
    
    # Load model
    print("\n[3/3] Loading Aurora 0.25° Fine-Tuned model...")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)
    print("  ✓ Model loaded")
    
    # Process each init hour
    total_runs = len(args.init_hours)
    total_start_time = time.time()
    
    for run_idx, init_hour in enumerate(args.init_hours, start=1):
        run_start_time = time.time()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n  [{timestamp}] [{run_idx}/{total_runs}] {args.date} init {init_hour:02d}:00 UTC")
        
        try:
            # Prepare batch
            batch = prepare_batch(args.date, download_path, init_hour=init_hour)
            init_time = batch.metadata.time[0]
            print(f"    Init time: {init_time}")
            
            # Format init time for filenames
            init_dt = init_time
            date_fmt = init_dt.strftime("%Y%m%d")
            init_fmt = init_dt.strftime("%H%M")
            
            p = next(model.parameters())
            batch = batch.to(device).type(p.dtype)
            
            with torch.inference_mode():
                rollout_gen = rollout(model, batch, steps=args.num_steps)
                
                for step, pred in enumerate(rollout_gen, start=1):
                    pred_cpu = pred.to("cpu")
                    
                    # Keep 720 lat points (no regrid) - matches WB2 Aurora reference
                    lead_hours = step * 6
                    pred_ds = batch_to_dataset(pred_cpu, step)
                    out_path = output_dir / f"aurora_pred_{date_fmt}_{init_fmt}_step{step:02d}_{lead_hours:03d}h.nc"
                    
                    pred_ds.to_netcdf(out_path)
                    print(f"    Step {step:2d} (+{lead_hours:3d}h) -> {out_path.name}")
                    
                    del pred_ds, pred_cpu
                    gc.collect()
            
            del batch
            torch.cuda.empty_cache()
            gc.collect()
            
            run_elapsed = time.time() - run_start_time
            print(f"    ✓ Done in {run_elapsed:.1f}s")
            
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Output saved to: {output_dir}")
    nc_files = list(output_dir.glob('*.nc'))
    print(f"  Files: {len(nc_files)} .nc files")


if __name__ == "__main__":
    main()
