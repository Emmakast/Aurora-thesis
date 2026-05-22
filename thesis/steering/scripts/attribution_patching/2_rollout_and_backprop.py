"""
2_rollout_and_backprop.py

Goal: Load the neutral state, run a manual temporal rollout for N steps, 
calculate a differentiable phenomenon index (spatial loss) directly on Aurora outputs, 
and backpropagate to get the 3D gradients on the intermediate latent.

NOTE: The rollout logic mirrors Aurora's own rollout.py exactly:
  1. batch_transform_hook  — adjusts variables
  2. batch.type(dtype)     — casts to model dtype
  3. batch.crop(patch_size) — crops 721→720 to be divisible by patch size
  4. dataclasses.replace    — proper batch re-wrapping with history concatenation
"""
import dataclasses
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "data_loader"))
from extract_latents_hres import prepare_batch, download_data, download_static
from aurora import Aurora, Batch, Metadata
import xarray as xr

def get_aurora_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    # We must be in eval mode to prevent BatchNorm/Dropout stochasticity,
    # but we DO NOT use torch.inference_mode() so gradients still flow.
    model.eval()
    model = model.to(device)
    return model

def load_dates(csv_path, phase):
    df = pd.read_csv(csv_path)
    df_phase = df[df['Type'] == phase]
    dates = []
    for _, row in df_phase.iterrows():
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    return dates

def load_single_neutral_state(date_str, device):
    """Load a single initial 3D spatial-temporal tensor for the given date."""
    download_path = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
    download_path.mkdir(parents=True, exist_ok=True)
    download_static(download_path)
    download_data(date_str, download_path)
        
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
    
    batch = prepare_batch(date_str, download_path, init_hour=12, static_vars_ds=static_vars_ds)
    batch = batch.to(device)
    return batch

def calculate_spatial_loss(prediction):
    """
    Calculates a purely differentiable spatial loss metric L on the raw high-resolution output.
    Extracts Mean Sea Level Pressure (MSLP) and computes the index: (Mean Midlat) - (Mean Polar).
    
    NOTE: The grid may be cropped (e.g. 720x1440 instead of 721x1440) due to
    Aurora's batch.crop(patch_size). We derive the grid shape dynamically.
    """
    # 1. Extract MSLP from Aurora's output data structure
    # Shape is (B, T, H, W) — squeeze batch and time dims
    # CRITICAL: Cast to float32 because MSLP values (~101325 Pa) overflow float16 max (~65504)
    mslp = prediction.surf_vars['msl'].squeeze(0).squeeze(0).float()  # (H, W) in fp32
    H, W = mslp.shape
    
    # 2. Generate lat/lon grids that match the ACTUAL cropped dimensions
    lat_1d = torch.linspace(90, -90, H, device=mslp.device)
    lon_1d = torch.linspace(0, 360, W + 1, device=mslp.device)[:-1]
    lat_grid, _ = torch.meshgrid(lat_1d, lon_1d, indexing='ij')
    
    # 3. Create geographic masks
    polar_mask = lat_grid > 60.0
    midlat_mask = (lat_grid > 30.0) & (lat_grid <= 60.0)
    
    # 4. Pure PyTorch tensor math to ensure gradient flow
    mean_polar = (mslp * polar_mask).sum() / (polar_mask.sum() + 1e-8)
    mean_midlat = (mslp * midlat_mask).sum() / (midlat_mask.sum() + 1e-8)
    
    # The "loss" is the index we want to maximize/attribute
    index = mean_midlat - mean_polar
    return index

def main():
    # 1. Configuration
    target_layer_name = "backbone.encoder_layers.2"
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv"
    rollout_steps = 1  # 1 step = 6h forecast; more steps OOM even on H100
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model
    model = get_aurora_model()
    
    # 3. Register Forward Hook with retain_grad()
    saved_tensors = {}
    
    def hook_fn_backward_prep(module, input, output):
        # Extract the tensor
        tensor = output[0] if isinstance(output, tuple) else output
        
        # We only want to track gradients for the initial state (t=0)
        # So we check if we've already saved it
        if 'target_latent' not in saved_tensors:
            # Tell PyTorch to keep the gradient for this intermediate tensor
            tensor.retain_grad()
            saved_tensors['target_latent'] = tensor

    # Register hook on the original module BEFORE checkpointing wrappers rename it
    target_layer = dict([*model.named_modules()])[target_layer_name]
    hook_handle = target_layer.register_forward_hook(hook_fn_backward_prep)

    # 3.5 Enable Activation Checkpointing
    # CRITICAL: Enable activation checkpointing to save memory during backpropagation.
    # This must be done AFTER registering the hook so the hook attaches to the real module, 
    # but the wrappers will still trigger it when they call the underlying module.
    model.configure_activation_checkpointing()

    # 4. Temporal Rollout — mirrors Aurora's own rollout.py exactly
    print(f"Running temporal rollout for {rollout_steps} steps...")
    
    # Pick the first Neutral date for this run
    neutral_dates = load_dates(dates_csv, phase="Neutral")
    target_date = neutral_dates[0]
    print(f"Using Neutral date: {target_date}")
    
    batch = load_single_neutral_state(target_date, device)
    
    # Apply the SAME preprocessing that Aurora's rollout() does:
    #   1. batch_transform_hook — adjusts available variables
    #   2. type(dtype)          — casts to model's parameter dtype
    #   3. crop(patch_size)     — crops grid to be divisible by patch size (721→720)
    p = next(model.parameters())
    batch = model.batch_transform_hook(batch)
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    
    for step in range(rollout_steps):
        print(f"  Forward pass step {step+1}/{rollout_steps}...")
        pred = model.forward(batch)
        
        # Re-wrap using dataclasses.replace, exactly like Aurora's rollout.py
        # This concatenates history (dropping oldest timestep) for autoregressive input
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
        
    final_prediction = pred  # Use the last prediction, not the re-wrapped batch
    
    # 5. Calculate Loss
    print("Calculating spatial index (loss) on raw high-res outputs...")
    loss = calculate_spatial_loss(final_prediction)
    print(f"Loss (AO Index Proxy): {loss.item()}")
    
    # 6. Backpropagation
    print("Backpropagating through the rollout...")
    loss.backward()
    
    # 7. Extract and Save Gradients
    gradient_tensor = saved_tensors['target_latent'].grad
    
    if gradient_tensor is None:
        raise ValueError("Gradient was not computed. Check if the computation graph was broken.")
        
    print(f"Gradient captured! Shape: {gradient_tensor.shape}")
    
    # Move to CPU for saving to save VRAM footprint
    gradient_tensor = gradient_tensor.detach().cpu()
    
    torch.save(gradient_tensor, output_dir / f"neutral_gradients_{target_date}.pt")
    hook_handle.remove()
    print("Done!")

if __name__ == "__main__":
    main()

