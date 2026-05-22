#!/usr/bin/env python3
"""
Visualize steered vs non-steered Aurora predictions for polar vortex steering.

Compares:
- Base prediction (alpha=0.0, no steering)
- Steered prediction (alpha=...>0, steered)
- Difference (steered - base)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import argparse

# File paths
AURORA_DIR = Path("/home/ekasteleyn/aurora_thesis")
OUTPUT_DIR = Path(__file__).parent


def plot_comparison(base_ds: xr.Dataset, steered_ds: xr.Dataset, 
                    var: str, phenomenon: str, alpha: float, level: int = None, output_path: Path = None):
    """Plot base, steered, and difference maps for a variable."""
    
    # Extract data
    if level is not None and 'level' in base_ds[var].dims:
        base_field = base_ds[var].sel(level=level).values
        steered_field = steered_ds[var].sel(level=level).values
        title_suffix = f" @ {level} hPa"
    else:
        base_field = base_ds[var].values
        steered_field = steered_ds[var].values
        title_suffix = ""
    
    diff_field = steered_field - base_field
    
    lat = base_ds.latitude.values
    lon = base_ds.longitude.values
    lons2d, lats2d = np.meshgrid(lon, lat)
    
    # Setup figure and projection based on phenomenon
    data_proj = ccrs.PlateCarree()
    if phenomenon == "AO":
        proj = ccrs.NorthPolarStereo()
        extent = [-180, 180, 30, 90]
    else: # MJO / ENSO
        proj = ccrs.PlateCarree(central_longitude=180)
        extent = [-180, 180, -40, 40] # Tropical/Mid-lat focus
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), 
                             subplot_kw={"projection": proj})
    
    # Color ranges
    vmin = np.nanpercentile(base_field, 2)
    vmax = np.nanpercentile(base_field, 98)
    diff_max = np.nanpercentile(np.abs(diff_field), 98)
    
    fields = [base_field, steered_field, diff_field]
    titles = [f"Base (α=0.0){title_suffix}", 
              f"Steered (α={alpha}){title_suffix}", 
              f"Difference (Steered - Base){title_suffix}"]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    vmins = [vmin, vmin, -diff_max]
    vmaxs = [vmax, vmax, diff_max]
    
    for ax, field, title, cmap, vm, vM in zip(axes, fields, titles, cmaps, vmins, vmaxs):
        ax.set_extent(extent, crs=data_proj)
        im = ax.pcolormesh(lons2d, lats2d, field, cmap=cmap, vmin=vm, vmax=vM,
                           shading="auto", transform=data_proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, 
                     fraction=0.046, aspect=25)
    
    var_label = var.upper() if len(var) <= 3 else var
    fig.suptitle(f"{phenomenon} Steering: {var_label}{title_suffix}", fontsize=14, y=1.02)
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved {output_path}")
    
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize steered Aurora predictions.")
    parser.add_argument("--phenomenon", type=str, default="AO", choices=["AO", "MJO", "ENSO"], help="Phenomenon to visualize")
    parser.add_argument("--date", type=str, required=True, help="Date tag of the prediction (e.g., 20201202)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength alpha to compare against base")
    args = parser.parse_args()

    base_file = AURORA_DIR / f"base_{args.phenomenon.lower()}_{args.date}_alpha_0.0.nc"
    steered_file = AURORA_DIR / f"steered_{args.phenomenon.lower()}_{args.date}_alpha_{args.alpha}.nc"

    print("Loading datasets...")
    base_ds = xr.open_dataset(base_file)
    steered_ds = xr.open_dataset(steered_file)
    
    print(f"Base file: {base_file}")
    print(f"Steered file: {steered_file}")
    print(f"Variables: {list(base_ds.data_vars)}")
    
    # Plot Z at 50 hPa (polar vortex level / stratosphere)
    print("\nPlotting Z at 50 hPa...")
    plot_comparison(base_ds, steered_ds, 'z', args.phenomenon, args.alpha, level=50,
                    output_path=OUTPUT_DIR / f"{args.phenomenon.lower()}_steered_vs_base_z50_alpha_{args.alpha}.png")
    
    # Plot Z at 500 hPa (mid-troposphere)
    print("Plotting Z at 500 hPa...")
    plot_comparison(base_ds, steered_ds, 'z', args.phenomenon, args.alpha, level=500,
                    output_path=OUTPUT_DIR / f"{args.phenomenon.lower()}_steered_vs_base_z500_alpha_{args.alpha}.png")
    
    # Plot 2m temperature
    print("Plotting 2m temperature...")
    plot_comparison(base_ds, steered_ds, '2t', args.phenomenon, args.alpha,
                    output_path=OUTPUT_DIR / f"{args.phenomenon.lower()}_steered_vs_base_2t_alpha_{args.alpha}.png")
    
    # Plot MSLP
    print("Plotting mean sea level pressure...")
    plot_comparison(base_ds, steered_ds, 'msl', args.phenomenon, args.alpha,
                    output_path=OUTPUT_DIR / f"{args.phenomenon.lower()}_steered_vs_base_msl_alpha_{args.alpha}.png")
    
    # Print some statistics
    print("\n=== Difference Statistics ===")
    for var in ['z', 't', 'msl', '2t', 'u', 'v']:
        if var in base_ds and var in steered_ds:
            if 'level' in base_ds[var].dims:
                for lvl in [50, 500, 850]:
                    if lvl in base_ds.level.values:
                        diff = (steered_ds[var].sel(level=lvl) - 
                                base_ds[var].sel(level=lvl)).values
                        print(f"{var}@{lvl}: mean={np.mean(diff):.2f}, "
                              f"std={np.std(diff):.2f}, "
                              f"max|diff|={np.max(np.abs(diff)):.2f}")
            else:
                diff = (steered_ds[var] - base_ds[var]).values
                print(f"{var}: mean={np.mean(diff):.2f}, "
                      f"std={np.std(diff):.2f}, "
                      f"max|diff|={np.max(np.abs(diff)):.2f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
