#!/usr/bin/env python3
"""
Analyze and visualize contrastive attention weights for Madden-Julian Oscillation (MJO).

This script compares Perceiver cross-attention patterns between Active and Neutral
MJO phases to identify which physical levels receive increased/decreased attention.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze contrastive MJO attention weights"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="target_dates.csv",
        help="Path to target_dates.csv",
    )
    parser.add_argument(
        "--attn-dir",
        type=str,
        default="s3://ekasteleyn-aurora-predictions/aurora_hres_validation",
        help="Directory containing attention tensor files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mjo_contrastive_attention.png",
        help="Output filename for the heatmap",
    )
    parser.add_argument(
        "--local-cache",
        type=str,
        default=None,
        help="Local cache directory for downloaded S3 files (optional)",
    )
    return parser.parse_args()


def load_dates_from_csv(csv_path: str, phenomenon: str = "MJO"):
    """
    Load target dates CSV and filter for specified phenomenon.
    
    Returns:
        active_dates: List of (year, month, day) tuples for Active type
        neutral_dates: List of (year, month, day) tuples for Neutral type
    """
    df = pd.read_csv(csv_path)
    
    # Filter for MJO phenomenon
    mjo_df = df[df["Phenomenon"] == phenomenon].copy()
    
    if mjo_df.empty:
        raise ValueError(f"No rows found with Phenomenon == '{phenomenon}'")
    
    # Separate by Type
    active_df = mjo_df[mjo_df["Type"] == "Active"]
    neutral_df = mjo_df[mjo_df["Type"] == "Neutral"]
    
    active_dates = list(zip(active_df["Year"], active_df["Month"], active_df["Day"]))
    neutral_dates = list(zip(neutral_df["Year"], neutral_df["Month"], neutral_df["Day"]))
    
    print(f"Found {len(active_dates)} Active MJO dates")
    print(f"Found {len(neutral_dates)} Neutral MJO dates")
    
    return active_dates, neutral_dates


def get_attention_file_path(attn_dir: str, year: int, month: int, day: int, 
                            local_cache: str = None) -> str:
    """
    Construct the path to an attention tensor file.
    Handles both local paths and S3 URIs.
    """
    filename = f"attn_{year:04d}_{month:02d}_{day:02d}.pt"
    
    if attn_dir.startswith("s3://"):
        if local_cache:
            # Return local cache path (assumes files are pre-downloaded)
            return str(Path(local_cache) / filename)
        else:
            # For S3, we'll need to use s5cmd or boto3
            return f"{attn_dir}/{filename}"
    else:
        return str(Path(attn_dir) / filename)


def load_attention_tensor(file_path: str) -> torch.Tensor:
    """
    Load attention tensor from file, handling S3 paths if needed.
    """
    if file_path.startswith("s3://"):
        # Use s5cmd to download to temp location
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        
        result = subprocess.run(
            ["s5cmd", "cp", file_path, tmp_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {file_path}: {result.stderr}")
        
        tensor = torch.load(tmp_path, map_location="cpu", weights_only=True)
        Path(tmp_path).unlink()  # Clean up temp file
        return tensor
    else:
        return torch.load(file_path, map_location="cpu", weights_only=True)


def reduce_tensor_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce attention tensor to 2D [num_latents, num_physical_levels].
    
    Assumes the last two dimensions are [num_latents, num_physical_levels].
    All preceding dimensions (batch, heads, spatial patches, etc.) are averaged.
    
    Args:
        tensor: Attention tensor of shape [..., num_latents, num_physical_levels]
    
    Returns:
        2D tensor of shape [num_latents, num_physical_levels]
    """
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions, got {tensor.ndim}")
    
    if tensor.ndim == 2:
        # Already 2D, no reduction needed
        return tensor
    
    # Average over all dimensions except the last two
    # This handles any combination of batch, heads, spatial dims
    dims_to_reduce = list(range(tensor.ndim - 2))
    reduced = tensor.mean(dim=dims_to_reduce)
    
    assert reduced.ndim == 2, f"Expected 2D tensor after reduction, got {reduced.ndim}D"
    return reduced


def aggregate_attention_tensors(dates: list, attn_dir: str, 
                                local_cache: str = None) -> torch.Tensor:
    """
    Load and aggregate attention tensors for a list of dates.
    
    Args:
        dates: List of (year, month, day) tuples
        attn_dir: Directory containing attention files
        local_cache: Optional local cache directory
    
    Returns:
        Mean 2D tensor [num_latents, num_physical_levels] across all dates
    """
    if not dates:
        raise ValueError("No dates provided for aggregation")
    
    tensors_2d = []
    loaded_count = 0
    failed_files = []
    
    for year, month, day in dates:
        file_path = get_attention_file_path(attn_dir, year, month, day, local_cache)
        
        try:
            tensor = load_attention_tensor(file_path)
            
            # Log shape on first successful load
            if loaded_count == 0:
                print(f"  Original tensor shape: {tensor.shape}")
            
            # Reduce to 2D
            tensor_2d = reduce_tensor_to_2d(tensor)
            tensors_2d.append(tensor_2d)
            loaded_count += 1
            
        except Exception as e:
            failed_files.append((file_path, str(e)))
            continue
    
    if not tensors_2d:
        raise RuntimeError(f"Failed to load any tensors. Errors: {failed_files[:5]}")
    
    if failed_files:
        print(f"  Warning: Failed to load {len(failed_files)} files")
        for path, err in failed_files[:3]:
            print(f"    - {path}: {err}")
    
    print(f"  Successfully loaded {loaded_count}/{len(dates)} tensors")
    print(f"  Reduced tensor shape: {tensors_2d[0].shape}")
    
    # Stack and compute mean across all dates
    stacked = torch.stack(tensors_2d, dim=0)  # [num_dates, num_latents, num_physical_levels]
    mean_tensor = stacked.mean(dim=0)  # [num_latents, num_physical_levels]
    
    return mean_tensor


def create_physical_level_labels(num_levels: int) -> list:
    """
    Create labels for physical levels.
    
    Assumes: Surface variables first, then 13 pressure levels.
    Adjust this function based on actual model configuration.
    """
    # Standard 13 pressure levels (hPa) in ERA5/HRES
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    # Number of surface variables = total - pressure levels
    num_surface = num_levels - len(pressure_levels)
    
    if num_surface < 0:
        # Fewer levels than expected, just use indices
        return [str(i) for i in range(num_levels)]
    
    labels = [f"Sfc{i}" for i in range(num_surface)]
    labels.extend([f"{p}hPa" for p in pressure_levels])
    
    # Truncate or pad if needed
    if len(labels) > num_levels:
        labels = labels[:num_levels]
    elif len(labels) < num_levels:
        labels.extend([f"L{i}" for i in range(len(labels), num_levels)])
    
    return labels


def plot_contrastive_heatmap(contrastive_matrix: np.ndarray, 
                             output_path: str,
                             title: str = "MJO Contrastive Attention (Active - Neutral)"):
    """
    Create and save a heatmap visualization of the contrastive attention matrix.
    """
    num_latents, num_physical_levels = contrastive_matrix.shape
    
    # Create labels
    latent_labels = [f"Latent {i}" for i in range(num_latents)]
    physical_labels = create_physical_level_labels(num_physical_levels)
    
    # Determine symmetric color limits centered at 0
    abs_max = np.abs(contrastive_matrix).max()
    vmin, vmax = -abs_max, abs_max
    
    # Create figure
    fig_width = max(12, num_physical_levels * 0.5)
    fig_height = max(6, num_latents * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap
    sns.heatmap(
        contrastive_matrix,
        ax=ax,
        cmap="RdBu_r",  # Red for positive (increased attention), Blue for negative
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True if num_latents * num_physical_levels <= 100 else False,
        fmt=".3f" if num_latents * num_physical_levels <= 100 else "",
        xticklabels=physical_labels,
        yticklabels=latent_labels,
        cbar_kws={"label": "Attention Difference (Active - Neutral)"},
        linewidths=0.5,
        linecolor="gray",
    )
    
    ax.set_xlabel("Physical Level", fontsize=12)
    ax.set_ylabel("Latent Level", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nHeatmap saved to: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MJO Contrastive Attention Analysis")
    print("=" * 60)
    
    # Step 1 & 2: Load and filter dates
    print("\n[1/4] Loading target dates...")
    active_dates, neutral_dates = load_dates_from_csv(args.csv, phenomenon="MJO")
    
    # Step 3: Aggregate attention tensors
    print("\n[2/4] Aggregating Active MJO attention tensors...")
    mean_active = aggregate_attention_tensors(
        active_dates, args.attn_dir, args.local_cache
    )
    
    print("\n[3/4] Aggregating Neutral MJO attention tensors...")
    mean_neutral = aggregate_attention_tensors(
        neutral_dates, args.attn_dir, args.local_cache
    )
    
    # Step 4: Compute contrastive matrix
    print("\n[4/4] Computing contrastive attention matrix...")
    contrastive_matrix = mean_active - mean_neutral
    contrastive_np = contrastive_matrix.numpy()
    
    print(f"  Contrastive matrix shape: {contrastive_np.shape}")
    print(f"  Value range: [{contrastive_np.min():.4f}, {contrastive_np.max():.4f}]")
    print(f"  Mean: {contrastive_np.mean():.4f}, Std: {contrastive_np.std():.4f}")
    
    # Step 5: Visualize
    print("\n[5/5] Creating visualization...")
    plot_contrastive_heatmap(contrastive_np, args.output)
    
    # Also save the raw contrastive matrix
    npz_path = args.output.replace(".png", "_data.npz")
    np.savez(
        npz_path,
        contrastive_matrix=contrastive_np,
        mean_active=mean_active.numpy(),
        mean_neutral=mean_neutral.numpy(),
    )
    print(f"Raw data saved to: {npz_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
