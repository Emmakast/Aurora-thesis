#!/usr/bin/env python3
"""
Visualize steered vs non-steered Aurora predictions for polar vortex steering.

Compares:
- Base prediction (alpha=0.0, no steering)
- Steered prediction (alpha=1.0, steering toward active AO)
- Difference (steered - base)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# File paths
AURORA_DIR = Path("/gpfs/home5/ekasteleyn/aurora_thesis")
BASE_FILE = AURORA_DIR / "base_polar_vortex_alpha_0.0.nc"
STEERED_FILE = AURORA_DIR / "steered_polar_vortex_alpha_1.0.nc"
OUTPUT_DIR = Path(__file__).parent


def plot_comparison(base_ds: xr.Dataset, steered_ds: xr.Dataset, 
                    var: str, level: int = None, output_path: Path = None):
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
    
    # Setup figure
    proj = ccrs.NorthPolarStereo()
    data_proj = ccrs.PlateCarree()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), 
                             subplot_kw={"projection": proj})
    
    # Color ranges
    vmin = np.nanpercentile(base_field, 2)
    vmax = np.nanpercentile(base_field, 98)
    diff_max = np.nanpercentile(np.abs(diff_field), 98)
    
    fields = [base_field, steered_field, diff_field]
    titles = [f"Base (α=0.0){title_suffix}", 
              f"Steered (α=1.0){title_suffix}", 
              f"Difference (Steered - Base){title_suffix}"]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    vmins = [vmin, vmin, -diff_max]
    vmaxs = [vmax, vmax, diff_max]
    
    for ax, field, title, cmap, vm, vM in zip(axes, fields, titles, cmaps, vmins, vmaxs):
        ax.set_extent([-180, 180, 30, 90], crs=data_proj)
        im = ax.pcolormesh(lons2d, lats2d, field, cmap=cmap, vmin=vm, vmax=vM,
                           shading="auto", transform=data_proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, 
                     fraction=0.046, aspect=25)
    
    var_label = var.upper() if len(var) <= 3 else var
    fig.suptitle(f"Polar Vortex Steering: {var_label}{title_suffix}", fontsize=14, y=1.02)
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved {output_path}")
    
    plt.close(fig)
    return fig


def main():
    print("Loading datasets...")
    base_ds = xr.open_dataset(BASE_FILE)
    steered_ds = xr.open_dataset(STEERED_FILE)
    
    print(f"Base file: {BASE_FILE}")
    print(f"Steered file: {STEERED_FILE}")
    print(f"Variables: {list(base_ds.data_vars)}")
    
    # Plot Z at 50 hPa (polar vortex level)
    print("\nPlotting Z at 50 hPa...")
    plot_comparison(base_ds, steered_ds, 'z', level=50,
                    output_path=OUTPUT_DIR / "steered_vs_base_z50.png")
    
    # Plot Z at 500 hPa (mid-troposphere)
    print("Plotting Z at 500 hPa...")
    plot_comparison(base_ds, steered_ds, 'z', level=500,
                    output_path=OUTPUT_DIR / "steered_vs_base_z500.png")
    
    # Plot 2m temperature
    print("Plotting 2m temperature...")
    plot_comparison(base_ds, steered_ds, '2t',
                    output_path=OUTPUT_DIR / "steered_vs_base_2t.png")
    
    # Plot MSLP
    print("Plotting mean sea level pressure...")
    plot_comparison(base_ds, steered_ds, 'msl',
                    output_path=OUTPUT_DIR / "steered_vs_base_msl.png")
    
    # Print some statistics
    print("\n=== Difference Statistics ===")
    for var in ['z', 't', 'msl', '2t']:
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
