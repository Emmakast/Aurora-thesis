#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Test FP64 Precision Comparison

This script runs Aurora in fp64 mode for 1 date and 2 init times,
then compares the output to WeatherBench2 Aurora predictions to verify
whether any deviations are due to floating point precision.

WB2 Aurora predictions: gs://weatherbench2/datasets/aurora/2022-1440x721.zarr
"""

import gc
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
import numpy as np
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

from aurora import Aurora, Batch, Metadata

# Disable TF32 for maximum precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DOWNLOAD_PATH = Path.home() / "downloads" / "hres_t0"
OUTPUT_DIR = Path.home() / "downloads" / "aurora_fp64_test"
WB2_HRES_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
WB2_AURORA_URL = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"

# Test configuration
TEST_DATE = "2022-01-15"
INIT_HOURS = [0, 12]


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def download_data(day: str, download_path: Path):
    """Download HRES T0 data for a specific day."""
    download_path.mkdir(parents=True, exist_ok=True)
    
    ds = xr.open_zarr(fsspec.get_mapper(WB2_HRES_URL), chunks=None)
    
    if not (download_path / f"{day}-surface-level.nc").exists():
        print(f"    Downloading surface variables for {day}...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(download_path / f"{day}-surface-level.nc"))
    
    if not (download_path / f"{day}-atmospheric.nc").exists():
        print(f"    Downloading atmospheric variables for {day}...")
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
    """Download static variables from HuggingFace."""
    if not (download_path / "static.nc").exists():
        print("  Downloading static variables from HuggingFace...")
        path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
        with open(path, "rb") as f:
            static_vars = pickle.load(f)
        
        ds_static = xr.Dataset(
            data_vars={k: (["latitude", "longitude"], v) for k, v in static_vars.items()},
            coords={
                "latitude": ("latitude", np.linspace(90, -90, 721)),
                "longitude": ("longitude", np.linspace(0, 360, 1440, endpoint=False)),
            },
        )
        ds_static.to_netcdf(str(download_path / "static.nc"))


def prepare_batch(day: str, download_path: Path, init_hour: int = 12) -> Batch:
    """Prepare batch using fp64 tensors."""
    import pandas as pd
    
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")
    
    if init_hour == 12:
        time_indices = [1, 2]
        init_time_idx = 2
    elif init_hour == 0:
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prev_surf_ds = xr.open_dataset(download_path / f"{prev_day}-surface-level.nc", engine="netcdf4")
        prev_atmos_ds = xr.open_dataset(download_path / f"{prev_day}-atmospheric.nc", engine="netcdf4")
        
        def _prepare_init00(x_prev: np.ndarray, x_curr: np.ndarray) -> torch.Tensor:
            combined = np.stack([x_prev[3], x_curr[0]], axis=0)
            return torch.from_numpy(combined[None][..., ::-1, :].copy()).double()  # FP64
        
        batch = Batch(
            surf_vars={
                "2t": _prepare_init00(prev_surf_ds["2m_temperature"].values, surf_vars_ds["2m_temperature"].values),
                "10u": _prepare_init00(prev_surf_ds["10m_u_component_of_wind"].values, surf_vars_ds["10m_u_component_of_wind"].values),
                "10v": _prepare_init00(prev_surf_ds["10m_v_component_of_wind"].values, surf_vars_ds["10m_v_component_of_wind"].values),
                "msl": _prepare_init00(prev_surf_ds["mean_sea_level_pressure"].values, surf_vars_ds["mean_sea_level_pressure"].values),
            },
            static_vars={
                "z": torch.from_numpy(static_vars_ds["z"].values).double(),
                "slt": torch.from_numpy(static_vars_ds["slt"].values).double(),
                "lsm": torch.from_numpy(static_vars_ds["lsm"].values).double(),
            },
            atmos_vars={
                "t": _prepare_init00(prev_atmos_ds["temperature"].values, atmos_vars_ds["temperature"].values),
                "u": _prepare_init00(prev_atmos_ds["u_component_of_wind"].values, atmos_vars_ds["u_component_of_wind"].values),
                "v": _prepare_init00(prev_atmos_ds["v_component_of_wind"].values, atmos_vars_ds["v_component_of_wind"].values),
                "q": _prepare_init00(prev_atmos_ds["specific_humidity"].values, atmos_vars_ds["specific_humidity"].values),
                "z": _prepare_init00(prev_atmos_ds["geopotential"].values, atmos_vars_ds["geopotential"].values),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()).double(),
                lon=torch.from_numpy(surf_vars_ds.longitude.values).double(),
                time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[0],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
            ),
        )
        prev_surf_ds.close()
        prev_atmos_ds.close()
        return batch
    else:
        raise ValueError(f"init_hour must be 0 or 12, got {init_hour}")
    
    def _prepare(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x[time_indices][None][..., ::-1, :].copy()).double()  # FP64
    
    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values).double(),
            "slt": torch.from_numpy(static_vars_ds["slt"].values).double(),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values).double(),
        },
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()).double(),
            lon=torch.from_numpy(surf_vars_ds.longitude.values).double(),
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[init_time_idx],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )
    
    return batch


def batch_to_dataset_fp64(pred: Batch, step: int) -> xr.Dataset:
    """Convert Aurora Batch to xarray Dataset preserving fp64 precision."""
    lat = pred.metadata.lat.numpy()
    lon = pred.metadata.lon.numpy()
    levels = list(pred.metadata.atmos_levels)
    
    data_vars = {}
    
    for name, tensor in pred.surf_vars.items():
        arr = tensor.double().numpy()  # Ensure fp64
        arr = arr[0, 0]
        data_vars[name] = (["latitude", "longitude"], arr.astype(np.float64))
    
    for name, tensor in pred.atmos_vars.items():
        arr = tensor.double().numpy()  # Ensure fp64
        arr = arr[0, 0]
        data_vars[name] = (["level", "latitude", "longitude"], arr.astype(np.float64))
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "latitude": lat.astype(np.float64),
            "longitude": lon.astype(np.float64),
            "level": levels,
        },
    )
    ds.attrs["valid_time"] = str(pred.metadata.time[0])
    ds.attrs["step"] = step
    ds.attrs["lead_hours"] = step * 6
    ds.attrs["precision"] = "float64"
    
    return ds


def compare_with_wb2_aurora(my_pred: Batch, init_time: datetime, step: int = 1):
    """Compare our Aurora prediction with WB2 Aurora predictions."""
    
    # Load WB2 Aurora predictions
    print(f"\n  Loading WB2 Aurora predictions...")
    wb2_aurora = xr.open_zarr(fsspec.get_mapper(WB2_AURORA_URL), chunks=None)
    
    # WB2 Aurora uses init time as 'time' and prediction_timedelta for lead time
    # prediction_timedelta is in nanoseconds: 6h = 21600000000000 ns
    lead_hours = step * 6
    lead_ns = lead_hours * 3600 * 1e9
    
    init_str = init_time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"    Init time: {init_str}")
    print(f"    Lead time: {lead_hours}h")
    
    # Select the matching prediction
    try:
        wb2_pred = wb2_aurora.sel(time=init_str, prediction_timedelta=np.timedelta64(lead_hours, 'h'))
    except Exception as e:
        print(f"    Could not find matching WB2 prediction: {e}")
        return
    
    # Variable mapping: my names -> WB2 names
    var_mapping = {
        "2t": "2m_temperature",
        "10u": "10m_u_component_of_wind", 
        "10v": "10m_v_component_of_wind",
        "msl": "mean_sea_level_pressure",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "q": "specific_humidity",
        "z": "geopotential",
    }
    
    print("\n  Variable comparison (My FP64 Aurora vs WB2 Aurora):")
    print("  " + "-" * 70)
    print(f"  {'Var':6s} | {'Max Diff':12s} | {'Mean Diff':12s} | {'Rel Diff':12s} | Exact?")
    print("  " + "-" * 70)
    
    # Get my prediction arrays
    my_lat = my_pred.metadata.lat.numpy()
    my_lon = my_pred.metadata.lon.numpy()
    
    # WB2 Aurora has lat from 89.875 to -89.875 (N->S), lon 0 to 359.75
    wb2_lat = wb2_pred.latitude.values
    wb2_lon = wb2_pred.longitude.values
    
    for my_var, wb2_var in var_mapping.items():
        # Get my prediction
        if my_var in my_pred.surf_vars:
            my_arr = my_pred.surf_vars[my_var].numpy()[0, 0]  # (lat, lon)
        elif my_var in my_pred.atmos_vars:
            my_arr = my_pred.atmos_vars[my_var].numpy()[0, 0]  # (level, lat, lon)
        else:
            continue
        
        # Get WB2 prediction  
        wb2_var_clean = wb2_var.replace('\t', '')  # WB2 has tabs in some var names
        if wb2_var_clean not in wb2_pred and wb2_var not in wb2_pred:
            # Try both with and without tab
            possible_vars = [v for v in wb2_pred.data_vars if wb2_var_clean in v or my_var in v.lower()]
            if possible_vars:
                wb2_var = possible_vars[0]
            else:
                print(f"  {my_var:6s} | NOT IN WB2")
                continue
        
        wb2_arr = wb2_pred[wb2_var].values if wb2_var in wb2_pred else wb2_pred[wb2_var_clean].values
        
        # Need to align coordinates - check if lat order matches
        # My lat is from prepare_batch which flips: [::-1] so it's S->N
        # WB2 is N->S typically
        # Let's check
        if my_lat[0] < my_lat[-1]:  # My is S->N
            my_arr = my_arr[..., ::-1, :]  # Flip to N->S for comparison
        
        # Cast to float64 for comparison
        my_arr = my_arr.astype(np.float64)
        wb2_arr = wb2_arr.astype(np.float64)
        
        # Compute differences
        diff = my_arr - wb2_arr
        max_abs_diff = np.max(np.abs(diff))
        mean_abs_diff = np.mean(np.abs(diff))
        
        # Relative difference
        scale = np.mean(np.abs(wb2_arr)) + 1e-10
        rel_diff = mean_abs_diff / scale
        
        exact_match = np.allclose(my_arr, wb2_arr, rtol=0, atol=0)
        close_match = np.allclose(my_arr, wb2_arr, rtol=1e-5, atol=1e-5)
        
        status = "✓ EXACT" if exact_match else ("≈ close" if close_match else "✗ DIFF")
        
        print(f"  {my_var:6s} | {max_abs_diff:12.2e} | {mean_abs_diff:12.2e} | {rel_diff:12.2e} | {status}")


def main():
    import pandas as pd
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("  FP64 PRECISION TEST - Aurora vs WeatherBench2")
    print("=" * 70)
    print(f"  Test date: {TEST_DATE}")
    print(f"  Init hours: {INIT_HOURS}")
    print(f"  Device: {device}")
    print(f"  TF32 disabled: True")
    print(f"  Precision: float64")
    print("=" * 70)
    
    # Download static and test data
    print("\n[1/4] Downloading static variables...")
    download_static(DOWNLOAD_PATH)
    
    print("\n[2/4] Downloading HRES T0 data...")
    dates_to_download = {TEST_DATE}
    if 0 in INIT_HOURS:
        prev_day = (pd.to_datetime(TEST_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        dates_to_download.add(prev_day)
    for day in sorted(dates_to_download):
        download_data(day, DOWNLOAD_PATH)
    
    # Load model in fp64
    print("\n[3/4] Loading Aurora model (FP64)...")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.double()  # Convert to fp64
    model = model.to(device)
    print("  ✓ Model loaded in FP64")
    
    # Process each init hour
    print("\n[4/4] Running inference...")
    
    for init_hour in INIT_HOURS:
        print(f"\n  --- Init {init_hour:02d}:00 UTC ---")
        
        batch = prepare_batch(TEST_DATE, DOWNLOAD_PATH, init_hour=init_hour)
        init_time = batch.metadata.time[0]
        print(f"    Init time: {init_time}")
        
        batch = batch.to(device)
        
        # Run forward pass
        with torch.inference_mode():
            pred = model(batch)
        
        # Move to CPU and convert to dataset
        pred_cpu = pred.to("cpu")
        pred_ds = batch_to_dataset_fp64(pred_cpu, step=1)
        
        # Compute valid time (+6h)
        valid_dt = init_time + timedelta(hours=6)
        valid_time_str = valid_dt.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Save fp64 prediction
        date_fmt = init_time.strftime("%Y%m%d")
        init_fmt = init_time.strftime("%H%M")
        out_path = OUTPUT_DIR / f"aurora_fp64_{date_fmt}_{init_fmt}_step01.nc"
        
        # Save with explicit float64 encoding
        encoding = {var: {"dtype": "float64"} for var in pred_ds.data_vars}
        pred_ds.to_netcdf(out_path, encoding=encoding)
        print(f"    Saved: {out_path.name}")
        
        # Print dtype verification
        print(f"    Output dtypes:")
        for var in list(pred_ds.data_vars)[:3]:
            print(f"      {var}: {pred_ds[var].dtype}")
        
        # Compare with WB2 Aurora predictions
        compare_with_wb2_aurora(pred_cpu, init_time, step=1)
        
        del batch, pred, pred_cpu, pred_ds
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    print(f"  FP64 predictions saved to: {OUTPUT_DIR}")
    print(f"\n  Summary:")
    print(f"  - If differences are 0: Model is deterministic and WB2 used same setup")
    print(f"  - If differences are small (1e-6 to 1e-4): Likely FP32 vs FP64 precision")
    print(f"  - If differences are larger: Different model weights, inputs, or code version")


if __name__ == "__main__":
    main()
