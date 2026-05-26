"""
1_extract_and_mask_latents.py

Goal: Extract the "Clean" (Active) and "Corrupted" (Neutral) latents from the 
Aurora Swin Transformer model, calculate the mean difference (delta_v), 
and apply a geographic mask (e.g., polar region > 60°N).
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# --- Imports for Aurora Model & Dataloader ---
# Add data_loader directory to path to import extract_latents_hres
sys.path.append(str(Path(__file__).parent.parent / "data_loader"))
from extract_latents_hres import prepare_batch, download_data, download_static
from aurora import Aurora, rollout
import xarray as xr

def get_aurora_model():
    """Load the Aurora model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)
    return model

def load_dates(csv_path, phase):
    """Load target dates for Active or Neutral phases."""
    df = pd.read_csv(csv_path)
    df_phase = df[df['Type'] == phase]
    dates = []
    for _, row in df_phase.iterrows():
        # format as YYYY-MM-DD
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    return dates

def process_dates(dates, model, device, static_vars_ds, extracted_latents, download_path):
    latents_list = []
    for i, day in enumerate(dates):
        print(f"  [{i+1}/{len(dates)}] Processing {day}...", flush=True)
        # Load and prepare one batch at a time
        download_data(day, download_path)
        batch = prepare_batch(day, download_path, init_hour=12, static_vars_ds=static_vars_ds)
        batch = batch.to(device)
        
        # Run forward pass
        for pred in rollout(model, batch, steps=1):
            pass
            
        latent = extracted_latents['latent']
        num_nans = torch.isnan(latent).sum().item()
        num_infs = torch.isinf(latent).sum().item()
        
        if num_nans > 0 or num_infs > 0:
            print(f"    WARNING: Latent contains {num_nans} NaNs and {num_infs} Infs! Skipping {day} to prevent corruption.", flush=True)
            del batch
            torch.cuda.empty_cache()
            continue
            
        latents_list.append(latent)
        print(f"    ✓ Latent captured, shape: {latent.shape}", flush=True)
        
        # Free memory immediately
        del batch
        torch.cuda.empty_cache()
        
    if len(latents_list) == 0:
        raise ValueError("All dates resulted in NaN/Inf latents! Check input data.")
        
    return latents_list

def create_polar_mask(lat_grid, threshold=60.0):
    """
    Creates a geographic mask to isolate specific regions.
    Args:
        lat_grid: 1D array of latitude values corresponding to the spatial tokens.
        threshold: Latitude threshold (e.g., > 60.0 for Polar region).
    Returns:
        Boolean mask tensor.
    """
    mask = lat_grid > threshold
    return mask

def main():
    # 1. Configuration
    # We use backbone.encoder_layers.2 as this is exactly how it is named in PyTorch dict([*model.named_modules()])
    target_layer_name = "backbone.encoder_layers.2"  
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv"
    output_dir = Path("/scratch-shared/ekasteleyn/aurora_thesis_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_path = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
    download_path.mkdir(parents=True, exist_ok=True)
    download_static(download_path)
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model
    model = get_aurora_model()

    # 3. Setup Forward Hook
    extracted_latents = {}
    
    def hook_fn(module, input, output):
        # Detach and move to CPU, cast to float32 immediately to prevent FP16 overflow during accumulation
        if isinstance(output, tuple):
            extracted_latents['latent'] = output[0].detach().cpu().to(torch.float32)
        else:
            extracted_latents['latent'] = output.detach().cpu().to(torch.float32)

    # Register the hook on the target layer
    target_layer = dict([*model.named_modules()])[target_layer_name]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # 4. Extract Active (Clean) Latents
    active_dates = load_dates(dates_csv, phase="Active")
    print(f"Running forward passes for {len(active_dates)} Active dates...")
    
    # Strict FP32 / No grad needed for extraction to preserve memory
    with torch.inference_mode(): 
        active_latents_list = process_dates(
            active_dates, model, device, static_vars_ds, extracted_latents, download_path
        )
    
    # Calculate Mean Active Latent
    mean_active = torch.stack(active_latents_list).mean(dim=0)
    
    # 5. Extract Neutral (Corrupted) Latents
    neutral_dates = load_dates(dates_csv, phase="Neutral")
    print(f"Running forward passes for {len(neutral_dates)} Neutral dates...")
    with torch.inference_mode():
        neutral_latents_list = process_dates(
            neutral_dates, model, device, static_vars_ds, extracted_latents, download_path
        )
            
    # Calculate Mean Neutral Latent
    mean_neutral = torch.stack(neutral_latents_list).mean(dim=0)

    # 6. Calculate Delta V
    print("Calculating delta_v...")
    delta_v = mean_active - mean_neutral
    delta_v = delta_v.squeeze(0)
    
    # 7. Apply Geographic Mask with Dynamic Reshaping
    print("Applying geographic mask dynamically based on sequence length...")
    actual_num_tokens = delta_v.shape[0]
    
    # Dynamically calculate the grid shape since Aurora is a Swin U-Net 
    # and downsamples spatially by 2x2 at every encoder layer.
    # The global Earth grid maintains a 1:2 aspect ratio (Lat:Lon)
    lat_size = int(np.sqrt(actual_num_tokens / 2))
    lon_size = lat_size * 2
    print(f"Dynamic grid shape calculated: Lat={lat_size}, Lon={lon_size}")
    
    lat_1d = torch.linspace(90, -90, lat_size)
    lon_1d = torch.linspace(0, 360, lon_size + 1)[:-1]
    lat_grid, _ = torch.meshgrid(lat_1d, lon_1d, indexing='ij')
    lat_tokens = lat_grid.flatten()  # Shape: [actual_num_tokens]

    mask = create_polar_mask(lat_tokens, threshold=60.0)
    
    # Apply mask (broadcasting across channel dimension)
    # mask shape: [Tokens], delta_v shape: [Tokens, Channels]
    masked_delta_v = delta_v * mask.unsqueeze(1).to(delta_v.dtype)

    # 8. Save Tensors
    hook_handle.remove()
    print("Saving tensors to disk...")
    
    # Save to a temporary file first, then rename (atomic save)
    # This prevents corrupted .pt files if the script crashes during save
    tmp_path_delta = output_dir / "delta_v.pt.tmp"
    final_path_delta = output_dir / "delta_v.pt"
    torch.save(masked_delta_v, tmp_path_delta)
    os.rename(tmp_path_delta, final_path_delta)
    
    tmp_path_neutral = output_dir / "mean_neutral_latent.pt.tmp"
    final_path_neutral = output_dir / "mean_neutral_latent.pt"
    torch.save(mean_neutral, tmp_path_neutral)
    os.rename(tmp_path_neutral, final_path_neutral)
    
    print("Done!")

if __name__ == "__main__":
    main()
