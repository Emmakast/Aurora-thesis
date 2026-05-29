#!/usr/bin/env python3
"""
Visualize a grid of steered Aurora predictions across multiple alphas (Global View).
Row 1: Raw Predictions
Row 2: Differences (Steered - Base)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import argparse
import torch

AURORA_DIR = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/results")

def plot_alpha_grid_global(base_ds, steered_ds_dict, var, alphas, phenomenon, level=None, output_path=None, base_file=None):
    if level is not None and 'level' in base_ds[var].dims:
        base_field = base_ds[var].sel(level=level).values
        title_suffix = f" @ {level} hPa"
    else:
        base_field = base_ds[var].values
        title_suffix = ""

    lat, lon = base_ds.latitude.values, base_ds.longitude.values
    lons2d, lats2d = np.meshgrid(lon, lat)

    data_proj = ccrs.PlateCarree()
    # Always use a global projection
    proj = ccrs.PlateCarree(central_longitude=180)
    extent = [-180, 180, -90, 90]

    cols = len(alphas) + 1 # +1 for Base (alpha=0)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8), subplot_kw={"projection": proj})

    if np.isnan(base_field).all():
        file_msg = f" in {base_file}" if base_file else ""
        print(f"Warning: {var.upper()}{title_suffix}{file_msg} contains only NaNs.")
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = np.nanpercentile(base_field, 2), np.nanpercentile(base_field, 98)
    
    max_diff = 1.0
    for a in alphas:
        if a in steered_ds_dict:
            sf = steered_ds_dict[a][var].sel(level=level).values if level else steered_ds_dict[a][var].values
            if np.isnan(sf).all():
                print(f"Warning: {var.upper()}{title_suffix} in steered file for alpha={a} contains only NaNs.")
            diff_abs = np.abs(sf - base_field)
            if not np.isnan(diff_abs).all():
                max_diff = max(max_diff, np.nanpercentile(diff_abs, 98))

    def draw_map(ax, field, cmap, vm, vM, title):
        ax.set_global()
        im = ax.pcolormesh(lons2d, lats2d, field, cmap=cmap, vmin=vm, vmax=vM, transform=data_proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_title(title, fontsize=10)
        return im

    # Base Column (Alpha = 0)
    draw_map(axes[0, 0], base_field, "viridis", vmin, vmax, "Base (α=0)")
    draw_map(axes[1, 0], np.zeros_like(base_field), "RdBu_r", -max_diff, max_diff, "Diff (Base - Base)")

    im_raw = None
    im_diff = None

    # Steered Columns
    for i, a in enumerate(alphas):
        col = i + 1
        if a not in steered_ds_dict:
            axes[0, col].set_title(f"Missing α={a}")
            axes[1, col].set_title(f"Missing α={a}")
            continue

        sf = steered_ds_dict[a][var].sel(level=level).values if level else steered_ds_dict[a][var].values
        diff = sf - base_field

        im_raw = draw_map(axes[0, col], sf, "viridis", vmin, vmax, f"Steered (α={a})")
        im_diff = draw_map(axes[1, col], diff, "RdBu_r", -max_diff, max_diff, f"Diff (α={a} - Base)")

    if im_raw is not None:
        fig.colorbar(im_raw, ax=axes[0, :].ravel().tolist(), orientation="vertical", pad=0.02, shrink=0.8)
    if im_diff is not None:
        fig.colorbar(im_diff, ax=axes[1, :].ravel().tolist(), orientation="vertical", pad=0.02, shrink=0.8)

    fig.suptitle(f"{phenomenon} Steering Grid (Global): {var.upper()}{title_suffix}", fontsize=16)
    
    fig.subplots_adjust(top=0.92, bottom=0.05, left=0.02, right=0.92, wspace=0.1, hspace=0.1)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)

def plot_steering_vector(vector_path: Path, output_path: Path | None = None):
    vec = torch.load(vector_path, map_location="cpu")
    if vec.ndim == 3:
        # [B, tokens, C] -> scalar score per token
        score = torch.norm(vec, dim=-1).squeeze(0).numpy()
        plt.figure(figsize=(14, 4))
        plt.plot(score, lw=0.8)
        plt.title(f"Steering vector norm: {vector_path.name}")
        plt.xlabel("Token index")
        plt.ylabel("L2 norm")
        plt.tight_layout()
    elif vec.ndim == 2:
        score = torch.norm(vec, dim=-1).numpy()
        plt.figure(figsize=(14, 4))
        plt.plot(score, lw=0.8)
        plt.title(f"Steering vector norm: {vector_path.name}")
        plt.xlabel("Token index")
        plt.ylabel("L2 norm")
        plt.tight_layout()
    else:
        score = vec.squeeze().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(score)
        plt.title(f"Steering vector: {vector_path.name}")
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Grid plot across alphas (Global)")
    parser.add_argument("--phenomenon", type=str, default="AO", choices=["AO", "AAO", "MJO", "ENSO", "NAO", "PNA"])
    parser.add_argument("--name-suffix", type=str, default="ao81_polar", help="Suffix appended to phenomenon (e.g., 'ao81_polar')")
    parser.add_argument("--date", type=str, default="20170308", help="Date tag (e.g., 20170308)")
    parser.add_argument("--init-hour", type=int, default=12, help="Initialization hour (e.g., 12 or 0)")
    parser.add_argument("--alphas", type=float, nargs='+', default=[-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--steering-vector", type=str, default=None, help="Optional .pt file to plot steering-vector norm")
    parser.add_argument("--steering-vector-output", type=str, default=None, help="Output PNG for steering-vector plot")
    parser.add_argument("--mask-tag", type=str, default="polar_north_lat60p0", help="Mask tag used in steered file names")
    parser.add_argument("--data-dir", type=str, default="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts", help="Directory containing .nc files")
    args = parser.parse_args()

    if args.steering_vector:
        plot_steering_vector(
            Path(args.steering_vector),
            Path(args.steering_vector_output) if args.steering_vector_output else None,
        )
        return

    suffix_str = args.name_suffix if args.name_suffix.startswith("_") or args.name_suffix == "" else f"_{args.name_suffix}"
    phenom_str = f"{args.phenomenon.lower()}{suffix_str}"
    init_tag = f"{args.init_hour:02d}00"
    data_dir = Path(args.data_dir)

    base_file_1 = data_dir / f"base_{phenom_str}_{args.date}_{init_tag}_alpha_0.0.nc"
    base_file_2 = data_dir / f"base_{args.phenomenon.lower()}_{args.date}_{init_tag}_alpha_0.0.nc"
    base_file_3 = data_dir / f"base_{phenom_str}_{args.date}_{init_tag}.nc"
    base_file_4 = data_dir / f"base_{args.date}_{init_tag}.nc"

    if base_file_1.exists():
        base_file = base_file_1
    elif base_file_2.exists():
        base_file = base_file_2
    elif base_file_3.exists():
        base_file = base_file_3
    elif base_file_4.exists():
        base_file = base_file_4
    else:
        print(f"Base file not found. Tried:\n  {base_file_1}\n  {base_file_2}\n  {base_file_3}\n  {base_file_4}")
        return

    base_ds = xr.open_dataset(base_file)
    steered_ds_dict = {}

    for a in args.alphas:
        # Check standard layout + possible absence of mask_tag
        f1 = data_dir / f"steered_{phenom_str}_{args.date}_{init_tag}_{args.mask_tag}_alpha_{a}.nc" if args.mask_tag else None
        f2 = data_dir / f"steered_{phenom_str}_{args.date}_{init_tag}_alpha_{a}.nc"
        
        target_f = f1 if (f1 and f1.exists()) else f2

        if target_f.exists():
            steered_ds_dict[a] = xr.open_dataset(target_f)
        else:
            print(f"Warning: File missing for alpha={a} (tried {f1} and {f2})")

    # Generate plots
    plot_alpha_grid_global(base_ds, steered_ds_dict, 'z', args.alphas, args.phenomenon, level=50,
                           output_path=data_dir / f"grid_global_{phenom_str}_{args.date}_{init_tag}_z50.png", base_file=base_file)
    plot_alpha_grid_global(base_ds, steered_ds_dict, 'z', args.alphas, args.phenomenon, level=500,
                           output_path=data_dir / f"grid_global_{phenom_str}_{args.date}_{init_tag}_z500.png", base_file=base_file)
    plot_alpha_grid_global(base_ds, steered_ds_dict, 'msl', args.alphas, args.phenomenon,
                           output_path=data_dir / f"grid_global_{phenom_str}_{args.date}_{init_tag}_msl.png", base_file=base_file)
    
if __name__ == "__main__":
    main()
