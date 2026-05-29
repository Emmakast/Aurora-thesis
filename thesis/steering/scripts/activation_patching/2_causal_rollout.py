"""
2_causal_rollout.py

Goal: Loop through a list of "Neutral" dates, run them normally (Base) and with 
the active patch injected (Steered), and save both forecasts to NetCDF for evaluation.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

try:
    import boto3
    from dotenv import load_dotenv
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

sys.path.append(str(Path("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader")))
try:
    from extract_latents_hres import prepare_batch, download_data, download_static, batch_to_dataset
    from aurora import Aurora, rollout
except ImportError:
    print("Warning: Could not import Aurora or dataloader functions.")

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
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    return dates

def get_s3_client():
    if not HAS_BOTO3:
        return None
    env_path = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    access_key = os.getenv('UVA_S3_ACCESS_KEY')
    secret_key = os.getenv('UVA_S3_SECRET_KEY')
    if access_key and secret_key:
        return boto3.client(
            's3',
            endpoint_url="https://ceph-gw.science.uva.nl:8000",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
    return None

def fetch_base_prediction_from_s3(s3_client, date_str, output_path, hhmm="0000"):
    """Attempt to download base prediction from S3 to avoid redundant rollouts."""
    if not s3_client:
        return False
        
    date_formatted = date_str.replace("-", "")
    filename = f"prediction_{date_formatted}_{hhmm}_hres.nc"
    s3_key = f"aurora_hres_validation/{filename}"
    bucket = "ekasteleyn-aurora-predictions"
    
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        print(f"    Found base prediction on S3: {s3_key}. Downloading...")
        s3_client.download_file(bucket, s3_key, str(output_path))
        return True
    except Exception:
        # File likely not found
        return False

def upload_base_prediction_to_s3(s3_client, local_path, date_str, hhmm="0000"):
    """Save base prediction to S3 so it can be retrieved in future runs."""
    if not s3_client:
        return
        
    date_formatted = date_str.replace("-", "")
    filename = f"prediction_{date_formatted}_{hhmm}_hres.nc"
    s3_key = f"aurora_hres_validation/{filename}"
    bucket = "ekasteleyn-aurora-predictions"
    
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        print(f"    ↑ Uploaded base prediction to S3: {s3_key}")
    except Exception as e:
        print(f"    Failed to upload base prediction to S3: {e}")

def make_patch_hook(mean_active_latent):
    """
    Creates a forward hook that physically replaces the polar tokens in the 
    neutral latent with those from the mean_active_latent.
    """
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        
        # args[2] contains the input resolution (C, H, W) to this layer
        C_in, H_in, W_in = args[2]
        
        # Determine the spatial dimensions of the output x based on actual token count
        num_tokens = x.shape[1]
        import math
        H_out = int(math.sqrt(num_tokens / 2))
        W_out = num_tokens // H_out
        
        # Create spatial mask correctly accounting for H, W layout
        lat_1d = torch.linspace(90, -90, H_out, device=x.device)
        lon_1d = torch.linspace(0, 360, W_out + 1, device=x.device)[:-1]
        lat_grid, _ = torch.meshgrid(lat_1d, lon_1d, indexing='ij')
        
        spatial_mask = lat_grid > 60.0 # [H_out, W_out]
        
        # Apply the boolean spatial mask over the 1D tokens
        mask = spatial_mask.flatten() # Shape: [Tokens]
        
        # Ensure active_latent is on the correct device and dtype
        m_active = mean_active_latent.to(dtype=x.dtype, device=x.device)
        
        # Broadcast mask to match [Batch, Tokens, Channels]
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
            
        # Hard replacement of tokens where mask is True
        new_x = torch.where(mask_expanded, m_active, x)
        
        if is_tuple:
            return (new_x,) + output[1:]
        return new_x
        
    return hook

def main():
    target_layer_name = "backbone.encoder_layers.2"  
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv"
    output_dir = Path("/scratch-shared/ekasteleyn/aurora_thesis_output/patched_rollouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_path = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
    download_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s3_client = get_s3_client()
    
    # 1. Load Mean Active Latent
    mean_latent_path = output_dir / "mean_active_latent.pt"
    if not mean_latent_path.exists():
        raise FileNotFoundError(f"Missing {mean_latent_path}. Run script 1 first.")
    
    mean_active_latent = torch.load(mean_latent_path, map_location='cpu', weights_only=True)
    
    # Ensure [1, Tokens, Channels]
    if mean_active_latent.dim() == 2:
        mean_active_latent = mean_active_latent.unsqueeze(0)
    
    # 2. Setup Base and Steered Neutral runs
    neutral_dates = load_dates(dates_csv, phase="Neutral")
    print(f"Running causal rollouts for {len(neutral_dates)} Neutral dates...")

    # Load model and static vars just once
    print("Initializing Aurora model locally...")
    download_static(download_path)
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
    model = get_aurora_model()
    target_layer = dict([*model.named_modules()])[target_layer_name]
    
    rollout_steps = 12 # 3 days (12 steps of 6 hours)
    
    for i, day in enumerate(neutral_dates):
        print(f"\n[{i+1}/{len(neutral_dates)}] Processing Neutral date: {day}")
        
        steps_to_save = [4, 8, 12]
        base_files = {s: output_dir / f"base_{day}_step{s:02d}.nc" for s in steps_to_save}
        patched_files = {s: output_dir / f"patched_{day}_step{s:02d}.nc" for s in steps_to_save}
        
        all_base_exist = all(p.exists() for p in base_files.values())
        all_patched_exist = all(p.exists() for p in patched_files.values())
        
        # Download input batch data
        download_data(day, download_path)
        # If init_hour=0, we need the previous day's data for the -6h history step
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        download_data(prev_day, download_path)
        
        # --- Base Run ---
        if all_base_exist:
            print(f"    Base files for {day} already exist locally. Skipping base rollout.")
        else:
            print("    Running Base Rollout...")
            batch = prepare_batch(day, download_path, init_hour=0, static_vars_ds=static_vars_ds)
            batch = batch.to(device)
            
            with torch.inference_mode():
                for step_idx, pred in enumerate(rollout(model, batch, steps=rollout_steps)):
                    actual_step = step_idx + 1
                    if actual_step in steps_to_save:
                        pred_cpu = pred.to("cpu")
                        base_ds = batch_to_dataset(pred_cpu, step=actual_step)
                        tmp_base = output_dir / f"base_{day}_step{actual_step:02d}.nc.tmp"
                        base_ds.to_netcdf(tmp_base)
                        os.rename(tmp_base, base_files[actual_step])
                        print(f"      Saved base step {actual_step}")
            
            del batch
            torch.cuda.empty_cache()
            print("    ✓ Base rollout completed and saved.")

        # --- Patched Run ---
        if all_patched_exist:
            print(f"    Patched files for {day} already exist. Skipping patched rollout.")
        else:
            print("    Running Patched Rollout (Causal Tracing)...")
            # We must re-prepare the batch to ensure it's clean
            batch = prepare_batch(day, download_path, init_hour=0, static_vars_ds=static_vars_ds)
            batch = batch.to(device)
            
            hook_handle = target_layer.register_forward_hook(
                make_patch_hook(mean_active_latent)
            )
            
            with torch.inference_mode():
                for step_idx, pred in enumerate(rollout(model, batch, steps=rollout_steps)):
                    actual_step = step_idx + 1
                    
                    if actual_step in steps_to_save:
                        pred_cpu = pred.to("cpu")
                        patched_ds = batch_to_dataset(pred_cpu, step=actual_step)
                        tmp_patched = output_dir / f"patched_{day}_step{actual_step:02d}.nc.tmp"
                        patched_ds.to_netcdf(tmp_patched)
                        os.rename(tmp_patched, patched_files[actual_step])
                        print(f"      Saved patched step {actual_step}")
                    
                    if step_idx == 0:
                        hook_handle.remove()
                        
            del batch
            torch.cuda.empty_cache()
            print("    ✓ Patched rollout completed and saved.")

if __name__ == "__main__":
    main()
