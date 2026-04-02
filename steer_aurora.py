import sys
import os
import torch
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

import tempfile
try:
    import boto3
    from dotenv import load_dotenv
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Add the steering script directory to path to import helpers
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/scripts/steering')
try:
    from extract_latents_hres import prepare_batch, batch_to_dataset, download_data, download_static
except ImportError:
    print("Warning: Could not import prepare_batch and batch_to_dataset from extract_latents_hres.py")
    sys.exit(1)

try:
    from aurora import Aurora, rollout
except ImportError:
    print("Warning: Could not import Aurora. Make sure the aurora environment is active.")

def main():
    print("Starting Contrastive Activation Addition (CAA) Steering Pipeline...")
    
    # ==========================================
    # Step 1: Compute the Steering Vector
    # ==========================================
    csv_path = "target_dates.csv"
    latents_dir = Path("/tmp/ekasteleyn/aurora_hres_latents") # Update this if needed, user didn't specify exactly
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Filter for AO
    ao_df = df[df['Phenomenon'] == 'AO']
    active_dates = ao_df[ao_df['Type'] == 'Active']
    neutral_dates = ao_df[ao_df['Type'] == 'Neutral']
    
# S3 Client setup
    s3_client = None
    if HAS_BOTO3:
        load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/scripts/steering/.env")
        access_key = os.getenv('UVA_S3_ACCESS_KEY')
        secret_key = os.getenv('UVA_S3_SECRET_KEY')
        if access_key and secret_key:
            s3_client = boto3.client(
                's3',
                endpoint_url="https://ceph-gw.science.uva.nl:8000",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print("S3 Client initialized.")
            
    def load_mean_latent(dates_df, layer='encoder_2', hhmm='0000'):
        tensors = []
        for _, row in dates_df.iterrows():
            date_str = f"{int(row['Year']):04d}{int(row['Month']):02d}{int(row['Day']):02d}"
            filename = f"latent_{date_str}_{hhmm}_{layer}.pt"
            
            # Check local first
            possible_paths = [
                Path(filename),
                Path("thesis/results") / filename,
                Path(os.environ.get('TMPDIR', '/tmp/ekasteleyn')) / "aurora_hres_latents" / filename
            ]
            
            file_path = None
            for p in possible_paths:
                if p.exists():
                    file_path = p
                    break
                    
            if file_path is None and s3_client is not None:
                # Try downloading from S3
                s3_key = f"aurora_hres_validation/{filename}"
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        s3_client.download_file("ekasteleyn-aurora-predictions", s3_key, tmp.name)
                        file_path = Path(tmp.name)
                except Exception as e:
                    # print(f"Could not download {filename} from S3: {e}")
                    pass
                    
            if file_path is None or not file_path.exists():
                print(f"Warning: Latent file {filename} not found locally or on S3.")
                continue
                
            try:
                t = torch.load(file_path, weights_only=True, map_location='cpu').float()
                tensors.append(t)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
            if str(file_path).startswith('/tmp') and 'tmp' in file_path.name:
                file_path.unlink() # cleanup named temp file
            
        if not tensors:
            print(f"Warning: No valid latent files found for {dates_df['Type'].iloc[0]}. Returning empty.")
            return None
            
        stacked = torch.stack(tensors, dim=0)
        return stacked.mean(dim=0)

        
    print(f"Loading Active latents ({len(active_dates)} dates)...")
    mean_active = load_mean_latent(active_dates)
    
    print(f"Loading Neutral latents ({len(neutral_dates)} dates)...")
    mean_neutral = load_mean_latent(neutral_dates)
    
    if mean_active is None or mean_neutral is None:
        print("Missing latents. Creating dummy delta_v for demonstration purposes.")
        # Dummy shape for Aurora Swin: [1, 4, H, W, C]
        delta_v = torch.zeros((1, 4, 18, 36, 1536))
    else:
        delta_v = mean_active - mean_neutral
        
    print(f"Steering vector (delta_v) shape: {delta_v.shape}")
    
    # Masking out all levels except Z=0
    # Assuming shape is (B, Z, Y, X, C) corresponding to ndim=5
    masked_delta_v = torch.zeros_like(delta_v)
    
    if delta_v.ndim == 5:
        # Assuming dim 1 is Z/depth
        masked_delta_v[:, 0, :, :, :] = delta_v[:, 0, :, :, :]
        print("Masked all levels except Z=0 (dimension 1).")
    elif delta_v.ndim == 4:
        masked_delta_v[0, :, :, :] = delta_v[0, :, :, :]
    else:
        print(f"Warning: Unexpected ndim {delta_v.ndim}. Unable to precisely mask Z dimension without understanding shape. Masking whole tensor.")
        masked_delta_v = delta_v
        
    # ==========================================
    # Step 2: Implement the Intervention Hook
    # ==========================================
    def make_intervention_hook(steering_vec, alpha=1.0):
        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            
            # Broadcast batch if needed, cast dtype and device
            s_vec = steering_vec.to(dtype=x.dtype, device=x.device)
            # Add scaled steering vector
            new_x = x + (alpha * s_vec)
            
            if is_tuple:
                return (new_x,) + output[1:]
            return new_x
        return hook
        
    # ==========================================
    # Step 3: Run the Steered Inference
    # ==========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Aurora Model on {device}...")
    
    # Needs Aurora installed
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)
    
    # Select a base Neutral date, let's just pick one from the CSV
    base_date = neutral_dates.iloc[0]
    base_day_str = f"{int(base_date['Year']):04d}-{int(base_date['Month']):02d}-{int(base_date['Day']):02d}"
    print(f"Selected Base Neutral Date: {base_day_str}")
    
    # Standard preparation
    download_dir = Path("/tmp/aurora_data")
    download_dir.mkdir(exist_ok=True)
    
    print("Downloading static/base data if needed...")
    download_static(download_dir)
    download_data(base_day_str, download_dir)
    
    print("Preparing batch...")
    batch = prepare_batch(base_day_str, download_dir, init_hour=12)
    batch = batch.to(device)
    
    alpha_val = 1.0
    hook_handle = model.backbone.encoder_layers[2].register_forward_hook(
        make_intervention_hook(masked_delta_v, alpha=alpha_val)
    )
    
    print(f"Running steered inference (alpha={alpha_val})...")
    with torch.inference_mode():
        # Single step rollout
        for pred in rollout(model, batch, steps=1):
            pred_batch = pred
            break
            
    pred_batch = pred_batch.to("cpu")
        
    # Remove hook when done
    hook_handle.remove()
    
    # Convert and save
    print("Converting prediction to xarray...")
    ds = batch_to_dataset(pred_batch, step=1)
    
    output_filename = f"steered_polar_vortex_alpha_{alpha_val}.nc"
    ds.to_netcdf(output_filename)
    print(f"Saved steered output to {output_filename}")
    print("Done!")

if __name__ == "__main__":
    main()
