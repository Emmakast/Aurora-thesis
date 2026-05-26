"""
3_compute_3d_attribution.py

Goal: Calculate the final attribution score by multiplying gradients and delta_v.
Project the dynamically downsampled 3D Swin latents back into a human-readable 
2D geographic map, and plot it using Cartopy.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

import pandas as pd

def load_dates(csv_path, phase):
    df = pd.read_csv(csv_path)
    df_phase = df[df['Type'] == phase]
    dates = []
    for _, row in df_phase.iterrows():
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    return dates

def main():
    # 1. Configuration
    output_dir = Path("/scratch-shared/ekasteleyn/aurora_thesis_output")
    plots_dir = Path("./plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv"
    
    # Pick the first Neutral date used in script 2
    neutral_dates = load_dates(dates_csv, phase="Neutral")
    target_date = neutral_dates[0]
    
    # 2. Load Tensors
    print("Loading delta_v and gradients...")
    try:
        delta_v = torch.load(output_dir / "delta_v.pt", map_location="cpu").squeeze()
        gradients = torch.load(output_dir / f"neutral_gradients_{target_date}.pt", map_location="cpu").squeeze()
    except FileNotFoundError:
        print("Error: Could not find required tensors. Run scripts 1 and 2 first.")
        # For demonstration, creating mock tensors if missing
        print("Using mock data for demonstration.")
        # Mocking a downsampled grid (e.g., after a few Swin layers)
        num_tokens = 32400 
        channels = 1536
        delta_v = torch.randn((num_tokens, channels), dtype=torch.float32)
        gradients = torch.randn((num_tokens, channels), dtype=torch.float32)

    print(f"Delta V shape: {delta_v.shape}")
    print(f"Gradients shape: {gradients.shape}")

    # --- Diagnostics before any math ---
    print(f"Delta V total non-zero values: {(delta_v != 0).sum().item()}")
    print(f"Gradients total non-zero values: {(gradients != 0).sum().item()}")

    # --- FIX: Handle NaNs in Aurora's internal padding tokens ---
    # Without this, any NaN values cause the colorbar scaling to break, resulting in a white map.
    delta_v = torch.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0)
    gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Calculate Attribution Score
    # Element-wise multiplication (First-order Taylor approximation)
    print("Computing attribution patching scores...")
    # Using strict FP32 math for accumulation to prevent precision issues
    attribution_3d = (gradients.to(torch.float32) * delta_v.to(torch.float32))
    
    # Sum across the channel dimension (dim=1) to get a single score per spatial token
    # Resulting shape: [Tokens]
    attribution_1d = attribution_3d.sum(dim=-1)
    
    # 4. Dynamic Reshape from 1D Sequence to 2D Geographic Grid
    # Aurora uses a flattened Swin transformer sequence where spatial dims depend on the layer.
    actual_num_tokens = attribution_1d.shape[0]
    
    # Dynamically calculate the grid shape since the Earth grid maintains a 1:2 aspect ratio
    lat_size = int(np.sqrt(actual_num_tokens / 2))
    lon_size = lat_size * 2
    grid_shape = (lat_size, lon_size)
    
    print(f"Dynamically reshaping from {attribution_1d.shape} to {grid_shape}...")
    attribution_2d = attribution_1d.view(*grid_shape)
    
    # Convert to numpy for plotting
    attribution_map = attribution_2d.detach().numpy()
    
    # --- Diagnostics ---
    non_zeros = (attribution_map != 0).sum()
    total_pixels = attribution_map.size
    print(f"Map diagnostics: min={attribution_map.min():.6e}, max={attribution_map.max():.6e}")
    print(f"Non-zero pixels: {non_zeros}/{total_pixels} ({(non_zeros/total_pixels)*100:.2f}%)")
    
    if non_zeros == 0:
        print("WARNING: The attribution map is entirely zero. Gradients or delta_v might be completely blank.")
    
    # 5. Plotting with Cartopy
    print("Generating geographic plot...")
    
    # Define lat/lon coordinates using the dynamically calculated grid shape
    lats = np.linspace(90, -90, lat_size)
    lons = np.linspace(0, 360, lon_size + 1)[:-1]
    
    fig = plt.figure(figsize=(12, 6))
    # Use Robinson projection for a global geographic view
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    
    # Set maximum limit for colorbar to handle extreme outliers common in gradients
    # Isolate non-zero pixels so the 99.9th percentile isn't skewed by the polar mask
    active_pixels = np.abs(attribution_map[(attribution_map != 0) & (~np.isnan(attribution_map))])
    if len(active_pixels) > 0:
        vmax = np.percentile(active_pixels, 99.9)
    else:
        vmax = 1e-8
        
    # Safety fallback if vmax still evaluates to 0
    if vmax == 0.0:
        vmax = np.max(active_pixels) if len(active_pixels) > 0 else 1e-8
        
    vmin = -vmax
    
    print(f"Colorbar scaling set to vmin={vmin:.6e}, vmax={vmax:.6e}")
    
    # Plot the 2D grid
    # transform=ccrs.PlateCarree() is required because our data is in standard Lat/Lon
    mesh = ax.pcolormesh(
        lons, lats, attribution_map, 
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r', 
        vmin=vmin, vmax=vmax,
        shading='auto'
    )
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Attribution Score (Gradient × Delta V)', fontsize=12)
    
    plt.title(f'Attribution Patching Map (Swin Latents Downsampled to {lat_size}x{lon_size})', fontsize=14, pad=10)
    
    # Save the plot
    plot_path = plots_dir / "attribution_map.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {plot_path}")
    
    # Uncomment to display plot if running interactively
    # plt.show()

if __name__ == "__main__":
    main()
