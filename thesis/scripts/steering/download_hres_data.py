#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Download HRES T0 data for Aurora latent extraction.

This script pre-downloads all required data so the GPU job can start
processing immediately without wasting GPU time on I/O.
"""

import argparse
import pickle
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import hf_hub_download

# Default paths
DOWNLOAD_PATH = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
WB2_HRES_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
WB2_ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"


def download_static(download_path: Path):
    """Download static variables from HuggingFace."""
    if (download_path / "static.nc").exists():
        print("  ✓ Static variables already cached")
        return
    
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
    ds_static.to_netcdf(str(download_path / "static.nc"), engine="h5netcdf")
    print("  ✓ Static variables downloaded")


def download_data(day: str, download_path: Path, ds: xr.Dataset, source: str = "HRES"):
    """Download HRES T0 or ERA5 data for a specific day."""
    surf_path = download_path / f"{day}-surface-level.nc"
    atmos_path = download_path / f"{day}-atmospheric.nc"
    
    # Check if already downloaded
    if surf_path.exists() and atmos_path.exists():
        print(f"  ✓ {day} already cached")
        return
    
    # Download surface-level variables
    if not surf_path.exists():
        print(f"  Downloading {day} surface variables ({source})...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(surf_path), engine="h5netcdf")
    
    # Download atmospheric variables
    if not atmos_path.exists():
        print(f"  Downloading {day} atmospheric variables ({source})...")
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(atmos_path), engine="h5netcdf")
    
    print(f"  ✓ {day} downloaded")


def main():
    parser = argparse.ArgumentParser(description="Download HRES T0 data for Aurora")
    parser.add_argument("--dates", nargs="+", required=True, help="Dates to download (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", type=str, default=None, help=f"Cache directory (default: {DOWNLOAD_PATH})")
    parser.add_argument("--include-prev-day", action="store_true", 
                        help="Also download previous day for each date (needed for init_hour=0)")
    args = parser.parse_args()
    
    download_path = Path(args.cache_dir) if args.cache_dir else DOWNLOAD_PATH
    download_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  HRES T0 DATA DOWNLOAD")
    print("=" * 60)
    print(f"  Dates: {len(args.dates)}")
    print(f"  Cache: {download_path}")
    print("=" * 60)
    
    # Build full list of dates to download
    dates_to_download = set(args.dates)
    if args.include_prev_day:
        for day in args.dates:
            prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            dates_to_download.add(prev_day)
    
    dates_to_download = sorted(dates_to_download)
    print(f"\n[1/3] Total dates to download: {len(dates_to_download)}")
    
    # Download static variables
    print("\n[2/3] Static variables...")
    download_static(download_path)
    
    # Open WB2 datasets once
    print("\n[3/3] HRES T0 / ERA5 data...")
    print("  Opening WeatherBench2 datasets...")
    ds_hres = xr.open_zarr(fsspec.get_mapper(WB2_HRES_URL), chunks=None)
    ds_era5 = xr.open_zarr(fsspec.get_mapper(WB2_ERA5_URL), chunks=None)
    
    for day in dates_to_download:
        year = int(day.split("-")[0])
        if year < 2016:
            download_data(day, download_path, ds_era5, source="ERA5")
        else:
            download_data(day, download_path, ds_hres, source="HRES")
    
    ds_hres.close()
    ds_era5.close()
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Files cached at: {download_path}")
    
    # Show disk usage
    total_size = sum(f.stat().st_size for f in download_path.glob("*.nc"))
    print(f"  Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
