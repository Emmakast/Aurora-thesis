#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Plot ERA5, FuXi, and Pangu predictions for 2020.
Compares temperature at 500 and 850 hPa for a selected date.
"""

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Dataset URLs
ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
FUXI_URL = "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr"
PANGU_URL = "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"

# Select a date in 2020 (available in all datasets)
INIT_TIME = pd.Timestamp("2020-06-15 00:00")
LEAD_HOURS = 240  # 10-day forecast
LEVEL = 500  # hPa


def main():
    print("=" * 70)
    print("  ERA5 / FuXi / Pangu Comparison - 2020")
    print("=" * 70)

    # Open datasets
    print("\nOpening datasets...")
    era5 = xr.open_zarr(fsspec.get_mapper(ERA5_URL), chunks=None)
    fuxi = xr.open_zarr(fsspec.get_mapper(FUXI_URL), chunks=None)
    pangu = xr.open_zarr(fsspec.get_mapper(PANGU_URL), chunks=None)
    print("  ✓ All datasets opened")

    # Valid time (what we're forecasting)
    valid_time = INIT_TIME + pd.Timedelta(hours=LEAD_HOURS)
    lead_td = np.timedelta64(LEAD_HOURS, "h")

    print(f"\n  Init time:  {INIT_TIME}")
    print(f"  Lead time:  {LEAD_HOURS}h")
    print(f"  Valid time: {valid_time}")
    print(f"  Level:      {LEVEL} hPa")

    # Get ERA5 ground truth at valid time
    print("\nLoading ERA5 (ground truth)...")
    era5_temp = era5["temperature"].sel(time=valid_time, level=LEVEL).compute()
    print(f"  ERA5 shape: {era5_temp.shape}, dtype: {era5_temp.dtype}")

    # Get FuXi forecast
    print("Loading FuXi forecast...")
    fuxi_temp = fuxi["temperature"].sel(
        time=INIT_TIME, prediction_timedelta=lead_td, level=LEVEL
    ).compute()
    print(f"  FuXi shape: {fuxi_temp.shape}, dtype: {fuxi_temp.dtype}")

    # Get Pangu forecast
    print("Loading Pangu forecast...")
    pangu_temp = pangu["temperature"].sel(
        time=INIT_TIME, prediction_timedelta=lead_td, level=LEVEL
    ).compute()
    print(f"  Pangu shape: {pangu_temp.shape}, dtype: {pangu_temp.dtype}")

    # Get coordinates
    lat_era5 = era5.latitude.values
    lon_era5 = era5.longitude.values
    lat_fuxi = fuxi.latitude.values
    lon_fuxi = fuxi.longitude.values
    lat_pangu = pangu.latitude.values
    lon_pangu = pangu.longitude.values

    # Convert to numpy and ensure consistent orientation (N->S)
    era5_data = era5_temp.values
    fuxi_data = fuxi_temp.values
    pangu_data = pangu_temp.values

    if lat_era5[0] < lat_era5[-1]:
        era5_data = era5_data[::-1, :]
        lat_era5 = lat_era5[::-1]
    if lat_fuxi[0] < lat_fuxi[-1]:
        fuxi_data = fuxi_data[::-1, :]
        lat_fuxi = lat_fuxi[::-1]
    if lat_pangu[0] < lat_pangu[-1]:
        pangu_data = pangu_data[::-1, :]
        lat_pangu = lat_pangu[::-1]

    # Compute errors (FuXi and Pangu vs ERA5)
    # Interpolate to same grid if needed (ERA5 is 1440x721, Pangu is 0.25deg=1440x721, FuXi is 1440x721)
    fuxi_error = fuxi_data - era5_data
    pangu_error = pangu_data - era5_data

    print(f"\n  FuXi RMSE:  {np.sqrt(np.nanmean(fuxi_error**2)):.3f} K")
    print(f"  Pangu RMSE: {np.sqrt(np.nanmean(pangu_error**2)):.3f} K")
    print(f"  FuXi MAE:   {np.nanmean(np.abs(fuxi_error)):.3f} K")
    print(f"  Pangu MAE:  {np.nanmean(np.abs(pangu_error)):.3f} K")

    # Create figure
    print("\nCreating plot...")
    fig = plt.figure(figsize=(16, 12))
    projection = ccrs.Robinson()

    # Temperature range for consistent colorbar
    vmin, vmax = 220, 280  # K at 500 hPa

    # Row 1: ERA5, FuXi, Pangu temperature fields
    ax1 = fig.add_subplot(2, 3, 1, projection=projection)
    ax1.set_title(f"ERA5 (Ground Truth)\n{valid_time}", fontsize=11)
    im1 = ax1.pcolormesh(
        lon_era5, lat_era5, era5_data,
        transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    ax1.coastlines(linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)

    ax2 = fig.add_subplot(2, 3, 2, projection=projection)
    ax2.set_title(f"FuXi ({LEAD_HOURS}h forecast)\nInit: {INIT_TIME}", fontsize=11)
    im2 = ax2.pcolormesh(
        lon_fuxi, lat_fuxi, fuxi_data,
        transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    ax2.coastlines(linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)

    ax3 = fig.add_subplot(2, 3, 3, projection=projection)
    ax3.set_title(f"Pangu ({LEAD_HOURS}h forecast)\nInit: {INIT_TIME}", fontsize=11)
    im3 = ax3.pcolormesh(
        lon_pangu, lat_pangu, pangu_data,
        transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    ax3.coastlines(linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3)

    # Add colorbar for temperature
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label("Temperature (K)", fontsize=10)

    # Row 2: Error maps
    err_vmax = 5  # K

    ax4 = fig.add_subplot(2, 3, 4, projection=projection)
    ax4.set_title("(placeholder)", fontsize=11)
    ax4.coastlines(linewidth=0.5)
    ax4.set_global()
    ax4.text(0.5, 0.5, f"Level: {LEVEL} hPa", transform=ax4.transAxes, 
             ha='center', va='center', fontsize=14)

    ax5 = fig.add_subplot(2, 3, 5, projection=projection)
    ax5.set_title(f"FuXi Error\nRMSE: {np.sqrt(np.nanmean(fuxi_error**2)):.2f} K", fontsize=11)
    im5 = ax5.pcolormesh(
        lon_fuxi, lat_fuxi, fuxi_error,
        transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-err_vmax, vmax=err_vmax
    )
    ax5.coastlines(linewidth=0.5)
    ax5.add_feature(cfeature.BORDERS, linewidth=0.3)

    ax6 = fig.add_subplot(2, 3, 6, projection=projection)
    ax6.set_title(f"Pangu Error\nRMSE: {np.sqrt(np.nanmean(pangu_error**2)):.2f} K", fontsize=11)
    im6 = ax6.pcolormesh(
        lon_pangu, lat_pangu, pangu_error,
        transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-err_vmax, vmax=err_vmax
    )
    ax6.coastlines(linewidth=0.5)
    ax6.add_feature(cfeature.BORDERS, linewidth=0.3)

    # Add colorbar for errors
    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.015, 0.35])
    cbar2 = fig.colorbar(im5, cax=cbar_ax2)
    cbar2.set_label("Error (K)", fontsize=10)

    plt.suptitle(
        f"Temperature at {LEVEL} hPa: ERA5 vs FuXi vs Pangu\n"
        f"Valid: {valid_time} ({LEAD_HOURS}h forecast from {INIT_TIME})",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    output_path = "era5_fuxi_pangu_comparison_2020.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Saved to: {output_path}")

    era5.close()
    fuxi.close()
    pangu.close()


if __name__ == "__main__":
    main()
