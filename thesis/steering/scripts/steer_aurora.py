import sys
import os
import argparse
import torch
import pandas as pd
import gc
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
    parser = argparse.ArgumentParser(description="Contrastive Activation Addition Steering")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0], help="List of steering strengths")
    parser.add_argument("--phenomenon", type=str, default="AO", choices=["AO", "MJO", "ENSO", "AAO"], help="Phenomenon to steer")
    parser.add_argument("--csv", type=str, default="target_dates.csv", help="Target dates CSV file")
    parser.add_argument("--neutral-csv", type=str, default=None, help="Optional separate CSV file for Neutral dates")
    parser.add_argument("--name-suffix", type=str, default="", help="Suffix to append to the output filename (e.g., '_ao81')")
    parser.add_argument("--steps", type=int, default=1, help="Number of rollout steps (e.g. 12 for 3 days)")
    args = parser.parse_args()
    
    suffix_str = args.name_suffix if args.name_suffix.startswith("_") or args.name_suffix == "" else f"_{args.name_suffix}"
    print(f"Starting Contrastive Activation Addition (CAA) Steering Pipeline ({args.phenomenon}, alphas={args.alphas})...")
    
    # ==========================================
    # Step 1: Compute the Steering Vector
    # ==========================================
    csv_path = args.csv
    latents_dir = Path("/tmp/ekasteleyn/aurora_hres_latents") # Update this if needed, user didn't specify exactly
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Filter for the chosen phenomenon
    phenom_df = df[df['Phenomenon'] == args.phenomenon]
    active_dates = phenom_df[phenom_df['Type'] == 'Active']
    
    if args.neutral_csv and os.path.exists(args.neutral_csv):
        neutral_df = pd.read_csv(args.neutral_csv)
        neutral_phenom_df = neutral_df[neutral_df['Phenomenon'] == args.phenomenon]
        neutral_dates = neutral_phenom_df[neutral_phenom_df['Type'] == 'Neutral']
        print(f"Loaded Neutral dates from separate CSV: {args.neutral_csv}")
    else:
        neutral_dates = phenom_df[phenom_df['Type'] == 'Neutral']
    
    print(f"Loaded CSV: {csv_path}")
    print(f"Found {len(active_dates)} Active dates and {len(neutral_dates)} Neutral dates.")
    
# S3 Client setup
    s3_client = None
    if HAS_BOTO3:
        load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
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
        # Use nanmean to ignore NaNs in the dataset, and zero out any fully-NaN results
        mean_val = torch.nanmean(stacked, dim=0)
        mean_val = torch.nan_to_num(mean_val, nan=0.0, posinf=0.0, neginf=0.0)
        return mean_val

        
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
        # Ensure extremely clean data
        delta_v = torch.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0)
        
    print(f"Steering vector (delta_v) shape: {delta_v.shape}, max= {torch.max(delta_v)}, min= {torch.min(delta_v)}, norm={torch.norm(delta_v)}")
    
    # Keep all Z levels for full 3D storm structure
    # Shape is (B, Z, Y, X, C) - we want steering across all vertical levels
    masked_delta_v = delta_v.clone()
    print(f"Keeping all Z levels for 3D structure (shape: {masked_delta_v.shape})")
        
    # ==========================================
    # Step 2: Implement the Intervention Hook (Normalized)
    # ==========================================
    def make_intervention_hook(steering_vec, alpha=1.0):
        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            
            # 1. Cast and move steering vector to match current activation
            s_vec = steering_vec.to(dtype=x.dtype, device=x.device)
            
            # 2. Calculate L2 norms
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            s_vec_norm = torch.norm(s_vec, p=2, dim=-1, keepdim=True)
            
            # Prevent division by zero
            s_vec_norm = torch.clamp(s_vec_norm, min=1e-6)
            
            # 3. Create a unit vector, then scale it relative to x's magnitude
            # alpha now represents a percentage of x's norm (e.g., alpha=0.1 is 10%)
            normalized_s_vec = (s_vec / s_vec_norm) * x_norm * alpha
            
            # 4. Inject
            new_x = x + normalized_s_vec
            
            if is_tuple:
                return (new_x,) + output[1:]
            return new_x
        return hook
        
    # ==========================================
    # Step 3: Prepare the Data
    # ==========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prioritize persistent scratch-shared so we don't redownload data on every new job
    shared_scratch = Path("/scratch-shared/ekasteleyn/aurora_data")
    if shared_scratch.parent.exists():
        download_dir = shared_scratch
    else:
        # Fallback to local job scratch node
        download_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "aurora_data"
        
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {download_dir} for data...")
    print("Downloading static/base data if needed...")
    download_static(download_dir)

    # Select a fallback base date from the CSV (using the first Neutral date)
    base_date = neutral_dates.iloc[0]
    base_day_str = f"{int(base_date['Year']):04d}-{int(base_date['Month']):02d}-{int(base_date['Day']):02d}"
    print(f"Selected Base Date: {base_day_str}")
    
    download_data(base_day_str, download_dir)
    print("Preparing batch...")
    batch = prepare_batch(base_day_str, download_dir, init_hour=12)
    date_tag = base_day_str.replace("-", "")

    # ==========================================
    # Step 4: Run the Steered Inference
    # ==========================================
    print(f"Loading Aurora Model on {device}...")
    
    # Needs Aurora installed
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)

    # Move to device (handle tuples or direct tensors)
    if isinstance(batch, tuple):
        batch = tuple(t.to(device) if hasattr(t, 'to') else t for t in batch)
    else:
        batch = batch.to(device)
    
    base_output_filename = f"base_{args.phenomenon.lower()}{suffix_str}_{date_tag}_alpha_0.0.nc"
    
    if not os.path.exists(base_output_filename):
        print(f"Running base inference (alpha=0.0) without hook for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                base_pred_batch = pred
                
        base_pred_batch = base_pred_batch.to("cpu")
        base_ds = batch_to_dataset(base_pred_batch, step=args.steps)
        
        tmp_base_filename = f"{base_output_filename}.tmp_base"
        base_ds.to_netcdf(tmp_base_filename)
        os.rename(tmp_base_filename, base_output_filename)
        print(f"Saved base output to {base_output_filename}")
    else:
        print(f"Base output {base_output_filename} already exists, skipping base inference.")
        
    for alpha_val in args.alphas:
        print(f"Applying hook with alpha={alpha_val}...")
        
        # NOTE: aurora batch tuples might have issues.
        # But base inference logic is identical to steer inference logic inside the hook
        hook_handle = model.backbone.encoder_layers[2].register_forward_hook(
            make_intervention_hook(masked_delta_v, alpha=alpha_val)
        )
        
        print(f"Running steered inference (alpha={alpha_val}) for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                pred_batch = pred
                
        pred_batch = pred_batch.to("cpu")
            
        # Remove hook when done
        hook_handle.remove()
        
        print(f"Converting prediction to xarray for alpha={alpha_val}...")
        ds = batch_to_dataset(pred_batch, step=args.steps)

        output_filename = f"steered_{args.phenomenon.lower()}{suffix_str}_{date_tag}_alpha_{alpha_val}.nc"
        ds.to_netcdf(output_filename)
        print(f"Saved steered output to {output_filename}")

    print("Done!")

if __name__ == "__main__":
    main()