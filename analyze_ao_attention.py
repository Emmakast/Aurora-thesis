#!/usr/bin/env python3
"""
Analyze and visualize contrastive attention weights for Arctic Oscillation (AO).

This script compares Perceiver cross-attention patterns between Active and Neutral
AO phases to identify where the model focuses differently during active events.
Focuses on the Arctic region (30°N to 90°N).
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass


# Configuration
S3_BUCKET = "ekasteleyn-aurora-predictions"
S3_FOLDER = "aurora_hres_validation"
S3_ENDPOINT = "https://ceph-gw.science.uva.nl:8000"
ENV_FILE = "/home/ekasteleyn/aurora_thesis/thesis/scripts/steering/.env"

TARGET_DATES_CSV = "target_dates.csv"
OUTPUT_FIGURE = "ao_contrastive_attention.png"
OUTPUT_FIGURE_ARCTIC = "ao_contrastive_attention_30N_90N.png"

# Physical level labels (surface variables + 13 pressure levels)
PHYSICAL_LEVEL_LABELS = [
    "2m_temp", "10m_u", "10m_v", "mslp",  # Surface variables
    "1000", "925", "850", "700", "600", "500",  # Pressure levels (hPa)
    "400", "300", "250", "200", "150", "100", "50"
]


def get_s3_client():
    """Initialize S3 client for UVA Ceph storage."""
    load_dotenv(ENV_FILE)
    
    if not HAS_BOTO3:
        raise ImportError("boto3 is required for S3 access. Install with: pip install boto3")
    
    access_key = os.getenv('UVA_S3_ACCESS_KEY')
    secret_key = os.getenv('UVA_S3_SECRET_KEY')
    
    if not access_key or not secret_key:
        raise ValueError(
            f"S3 credentials not found. Ensure UVA_S3_ACCESS_KEY and UVA_S3_SECRET_KEY "
            f"are set in {ENV_FILE}"
        )
    
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )


def load_attention_tensor(s3_client, year: int, month: int, day: int) -> torch.Tensor:
    """Load a PyTorch attention tensor from S3."""
    import tempfile
    
    filename = f"attn_{year:04d}{month:02d}{day:02d}_0000_perceiver_cross_attn.pt"
    s3_key = f"{S3_FOLDER}/{filename}"
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        s3_client.download_file(S3_BUCKET, s3_key, tmp.name)
        tensor = torch.load(tmp.name, map_location="cpu", weights_only=True)
    
    return tensor


def reduce_to_2d(tensor: torch.Tensor, arctic_only: bool = False) -> torch.Tensor:
    """
    Reduce attention tensor to 2D [num_latents, num_physical_levels].
    
    Dynamically handles tensors of various shapes by averaging all dimensions
    except the last two (assumed to be latents and physical levels).
    
    Args:
        tensor: Attention tensor of various shapes
        arctic_only: If True, apply latitude mask for Arctic region (30°N to 90°N)
    
    Expected shapes:
    - [batch, heads, latents, physical]: 4D
    - [spatial, heads, latents, physical]: 4D (perceiver cross-attn)
    - [batch, heads, spatial, latents, physical]: 5D
    """
    # Convert to float32 to avoid overflow during averaging (float16 can overflow)
    tensor = tensor.float()
    
    ndim = tensor.ndim
    
    # Check if we need to apply Arctic latitudinal mask (30°N to 90°N)
    if arctic_only and ndim >= 4:
        # Assuming spatial dim is the first dim or second dim (after batch)
        spatial_dim_idx = 1 if ndim == 5 else 0
        num_spatial = tensor.shape[spatial_dim_idx]
        
        # Infer grid based on typical Aurora resolutions
        # Latitude goes from 90°N (index 0) to 90°S (last index)
        if num_spatial == 1801 * 3600:  # 0.1 degree resolution
            # 90°N = index 0, 30°N = (90-30)/0.1 = 600
            lat_start, lat_end = 0, 601  # 90°N to 30°N
            lon_dim = 3600
        elif num_spatial == 721 * 1440:  # 0.25 degree resolution
            # 90°N = index 0, 30°N = (90-30)/0.25 = 240
            lat_start, lat_end = 0, 241  # 90°N to 30°N
            lon_dim = 1440
        elif num_spatial == 180 * 360:  # 1 degree resolution
            # 90°N = index 0, 30°N = (90-30)/1 = 60
            lat_start, lat_end = 0, 61   # 90°N to 30°N
            lon_dim = 360
        else:
            raise ValueError(f"Unknown spatial dimension size for Arctic masking: {num_spatial}")
            
        # Reshape to explicitly include lat/lon, slice latitudes, then flatten back
        if ndim == 4:
            tensor = tensor.reshape(num_spatial // lon_dim, lon_dim, *tensor.shape[1:])
            tensor = tensor[lat_start:lat_end, :, :, :, :]
        elif ndim == 5:
            tensor = tensor.reshape(tensor.shape[0], num_spatial // lon_dim, lon_dim, *tensor.shape[2:])
            tensor = tensor[:, lat_start:lat_end, :, :, :, :]

    if ndim == 2 or (arctic_only and ndim < 4):
        return tensor
    elif ndim == 3:
        return torch.nanmean(tensor, dim=0)
    elif ndim == 4:
        # Flatten first two dims and nanmean
        flat = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
        return torch.nanmean(flat, dim=0)
    elif ndim >= 5:
        # Flatten all but last 2 dims
        flat = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
        return torch.nanmean(flat, dim=0)
    else:
        raise ValueError(f"Unexpected tensor dimension: {ndim}")


def aggregate_attention_for_dates(
    dates: list[tuple[int, int, int]],
    s3_client,
    verbose: bool = True,
    arctic_only: bool = False
) -> torch.Tensor:
    """
    Load and aggregate attention tensors for a list of dates.
    
    Args:
        dates: List of (year, month, day) tuples
        s3_client: Initialized boto3 S3 client
        verbose: Print progress information
        arctic_only: Whether to mask for Arctic area (30°N - 90°N)
        
    Returns:
        Mean 2D tensor [num_latents, num_physical_levels] across all dates
    """
    tensors_2d = []
    
    for i, (year, month, day) in enumerate(dates):
        try:
            tensor = load_attention_tensor(s3_client, year, month, day)
            tensor_2d = reduce_to_2d(tensor, arctic_only=arctic_only)
            tensors_2d.append(tensor_2d)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(dates)} files...")
                
        except Exception as e:
            print(f"  Warning: Error loading {year:04d}-{month:02d}-{day:02d}: {e}")
    
    if not tensors_2d:
        raise RuntimeError("No attention tensors were successfully loaded")
    
    # Stack and compute mean across all dates (use nanmean to handle any NaN values)
    stacked = torch.stack(tensors_2d, dim=0)  # [num_dates, latents, physical]
    mean_tensor = torch.nanmean(stacked, dim=0)  # [latents, physical]
    
    if verbose:
        print(f"  Aggregated {len(tensors_2d)} tensors -> shape {tuple(mean_tensor.shape)}")
    
    return mean_tensor


def plot_attention_line_plots(
    mean_active: np.ndarray,
    mean_neutral: np.ndarray,
    output_path: str,
    physical_labels: list[str] | None = None,
    figsize: tuple[float, float] = (12, 10)
) -> None:
    """
    Create stacked line plots for attention analysis.
    
    Plot 1 (Top): Raw baseline attention (neutral) for atmospheric latents.
    Plot 2 (Bottom): Contrastive difference (active - neutral) for atmospheric latents.
    
    Args:
        mean_active: 2D array [num_latents, num_physical_levels]
        mean_neutral: 2D array [num_latents, num_physical_levels]
        output_path: Path to save the figure
        physical_labels: Labels for x-axis (physical levels)
        figsize: Figure size
    """
    num_latents, num_physical = mean_neutral.shape
    
    # Use provided physical labels or generate defaults
    if physical_labels is None or len(physical_labels) != num_physical:
        physical_labels = [f"Level {i}" for i in range(num_physical)]
    
    # Compute contrastive difference
    contrastive = mean_active - mean_neutral
    
    # Define all latent indices
    latent_indices = list(range(num_latents))
    
    # Colors and markers for each latent
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']  # Red, Blue, Orange, Green
    markers = ['D', 'o', 's', '^']  # Diamond, Circle, Square, Triangle
    latent_names = [f'Latent {i}' for i in range(num_latents)]
    
    # X positions
    x = np.arange(num_physical)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # ===== Plot 1: Raw Baseline Attention (Neutral) =====
    for i, latent_idx in enumerate(latent_indices):
        if latent_idx < num_latents:
            ax1.plot(x, mean_neutral[latent_idx], 
                     color=colors[i], marker=markers[i], markersize=6,
                     linewidth=2, label=latent_names[i])
    
    ax1.set_ylabel("Attention Weight", fontsize=12)
    ax1.set_title("Baseline Attention Distribution (Neutral AO Phase)", 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.5, num_physical - 0.5)
    
    # ===== Plot 2: Contrastive Difference =====
    for i, latent_idx in enumerate(latent_indices):
        if latent_idx < num_latents:
            ax2.plot(x, contrastive[latent_idx], 
                     color=colors[i], marker=markers[i], markersize=6,
                     linewidth=2, label=latent_names[i])
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel("Physical Levels", fontsize=12)
    ax2.set_ylabel("Attention Difference\n(Active - Neutral)", fontsize=12)
    ax2.set_title("Contrastive Attention: AO Active vs Neutral", 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks and labels
    ax2.set_xticks(x)
    ax2.set_xticklabels(physical_labels, rotation=45, ha='right', fontsize=10)
    
    # Add vertical separator between surface and pressure levels if applicable
    if num_physical >= 4:
        for ax in [ax1, ax2]:
            ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention line plots to: {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("AO Contrastive Attention Analysis")
    print("=" * 60)
    
    # Step 1: Load and filter target dates
    print("\n[1/5] Loading target dates...")
    df = pd.read_csv(TARGET_DATES_CSV)
    ao_df = df[df["Phenomenon"] == "AO"].copy()
    print(f"  Found {len(ao_df)} AO dates out of {len(df)} total")
    
    # Step 2: Separate Active and Neutral dates
    print("\n[2/5] Separating Active and Neutral phases...")
    active_dates = ao_df[ao_df["Type"] == "Active"][["Year", "Month", "Day"]]
    neutral_dates = ao_df[ao_df["Type"] == "Neutral"][["Year", "Month", "Day"]]
    
    active_list = list(active_dates.itertuples(index=False, name=None))
    neutral_list = list(neutral_dates.itertuples(index=False, name=None))
    
    print(f"  Active AO dates:  {len(active_list)}")
    print(f"  Neutral AO dates: {len(neutral_list)}")
    
    if not active_list or not neutral_list:
        raise ValueError("Need both Active and Neutral dates for contrastive analysis")
    
    # Initialize S3 client
    print("\n[3/6] Initializing S3 client...")
    s3_client = get_s3_client()
    print("  ✓ S3 client initialized")
    
    for mode, arctic_mask, out_fig in [
        ("Global", False, OUTPUT_FIGURE),
        ("Arctic-Region (30N-90N)", True, OUTPUT_FIGURE_ARCTIC)
    ]:
        print(f"\n--- Processing {mode} Attention ---")
        
        # Step 3: Aggregate attention tensors
        print(f"\n[4/6] Aggregating Active phase attention ({mode})...")
        mean_active = aggregate_attention_for_dates(active_list, s3_client, arctic_only=arctic_mask)
        
        print(f"\n[5/6] Aggregating Neutral phase attention ({mode})...")
        mean_neutral = aggregate_attention_for_dates(neutral_list, s3_client, arctic_only=arctic_mask)
        
        # Step 4: Compute contrastive matrix
        print(f"\n[6/6] Computing contrastive attention matrix ({mode})...")
        contrastive = mean_active - mean_neutral
        
        # Convert to numpy
        mean_active_np = mean_active.numpy()
        mean_neutral_np = mean_neutral.numpy()
        contrastive_np = contrastive.numpy()
        
        print(f"  Contrastive matrix shape: {contrastive_np.shape}")
        print(f"  Value range: [{np.nanmin(contrastive_np):.6f}, {np.nanmax(contrastive_np):.6f}]")
        
        # Step 5: Visualize
        print(f"\nCreating visualization ({mode})...")
        
        num_physical = contrastive_np.shape[1]
        if num_physical == len(PHYSICAL_LEVEL_LABELS):
            physical_labels = PHYSICAL_LEVEL_LABELS
        else:
            physical_labels = None  
        
        plot_attention_line_plots(
            mean_active_np,
            mean_neutral_np,
            out_fig,
            physical_labels=physical_labels
        )
        
        # Save numerical results
        suffix = "_arctic_region" if arctic_mask else ""
        results_path = f"ao_contrastive_attention{suffix}.npy"
        np.save(results_path, contrastive_np)
        print(f"Saved numerical results to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
