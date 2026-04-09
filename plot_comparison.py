import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_comparison(base_file, steered_file, variable='z', level=None, save_path='comparison_plot.png'):
    """
    Plots the base prediction vs steered prediction side-by-side. 
    Also plots the difference (Steered - Base).
    """
    # Load the datasets
    base_ds = xr.open_dataset(base_file)
    steer_ds = xr.open_dataset(steered_file)
    
    # Extract the target variable
    base_var = base_ds[variable]
    steer_var = steer_ds[variable]
    
    # If the variable has multiple levels (e.g., geopotential 'z'), select one
    # 50 hPa or 500 hPa are common for Polar Vortex/AO
    if level is not None and 'level' in base_var.coords:
        base_var = base_var.sel(level=level, method='nearest')
        steer_var = steer_var.sel(level=level, method='nearest')
        
    # Calculate difference
    diff_var = steer_var - base_var

    # Setup North Polar Stereo projection
    # Best for viewing Arctic Oscillation/Polar Vortex
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': proj})
    
    # Common features for all maps
    for ax in axes:
        ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree()) # Northern Hemisphere Focus
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # Determine common min/max for Base and Steered to share the colorbar
    vmin = min(base_var.min().values, steer_var.min().values)
    vmax = max(base_var.max().values, steer_var.max().values)
    
    cmap = 'coolwarm' if variable == 't' else 'viridis' # Coolwarm for Temperature, Viridis for Z

    # 1. Base Prediction
    p0 = axes[0].pcolormesh(
        base_var.longitude, base_var.latitude, base_var.values,
        transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
    )
    axes[0].set_title(f'Base Prediction ({variable})')
    
    # 2. Steered Prediction
    p1 = axes[1].pcolormesh(
        steer_var.longitude, steer_var.latitude, steer_var.values,
        transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
    )
    axes[1].set_title(f'Steered Prediction (alpha=1.0)')
    fig.colorbar(p1, ax=axes[:2], orientation='horizontal', shrink=0.8, pad=0.05, label=f'{variable} values')

    # 3. Difference (Steered - Base)
    # Re-center difference colormap around 0
    diff_max = np.abs(diff_var.values).max()
    p2 = axes[2].pcolormesh(
        diff_var.longitude, diff_var.latitude, diff_var.values,
        transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, shading='auto'
    )
    axes[2].set_title(f'Difference (Steered - Base)')
    fig.colorbar(p2, ax=axes[2], orientation='horizontal', shrink=0.8, pad=0.05, label=f'Delta {variable}')

    # Add overall title
    level_str = f" at {level}hPa" if level else " (Surface)"
    plt.suptitle(f"Effects of Contrastive Activation Addition (CAA) on {variable.upper()}{level_str}", fontsize=16)

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # First Plot: Geopotential at 50hPa (Polar Vortex specific)
    plot_comparison(
        base_file="base_polar_vortex_alpha_0.0.nc", 
        steered_file="steered_polar_vortex_alpha_1.0.nc", 
        variable="z", 
        level=50,
        save_path="comparison_geopotential_50hPa.png"
    )
    
    # Second Plot: 2m Temperature (Effects on Weather near ground)
    plot_comparison(
        base_file="base_polar_vortex_alpha_0.0.nc", 
        steered_file="steered_polar_vortex_alpha_1.0.nc", 
        variable="2t", 
        level=None,
        save_path="comparison_temperature_2m.png"
    )
