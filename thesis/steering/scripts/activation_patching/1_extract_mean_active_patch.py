"""
1_extract_mean_active_patch.py

Goal: Extract the mean latent state of the 3rd encoder (backbone.encoder_layers.2) 
exclusively for "Active" dates, and save it. Queries S3 first to avoid redundant extraction.
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

# Add data_loader directory to path
sys.path.append(str(Path("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader")))
try:
    from extract_latents_hres import prepare_batch, download_data, download_static
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

def fetch_latent_from_s3(s3_client, date_str, layer="encoder_2", hhmm="0000"):
    """Attempt to download latent from S3."""
    if not s3_client:
        return None
    
    date_formatted = date_str.replace("-", "")
    filename = f"latent_{date_formatted}_{hhmm}_{layer}.pt"
    s3_key = f"aurora_hres_validation/{filename}"
    bucket = "ekasteleyn-aurora-predictions"
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            s3_client.download_file(bucket, s3_key, tmp.name)
            tensor = torch.load(tmp.name, weights_only=True, map_location='cpu').float()
            os.unlink(tmp.name)
            return tensor
    except Exception:
        # File likely not found
        return None

def upload_latent_to_s3(s3_client, tensor, date_str, layer="encoder_2", hhmm="0000"):
    """Save latent to S3 so it can be retrieved in future runs."""
    if not s3_client:
        return
        
    date_formatted = date_str.replace("-", "")
    filename = f"latent_{date_formatted}_{hhmm}_{layer}.pt"
    s3_key = f"aurora_hres_validation/{filename}"
    bucket = "ekasteleyn-aurora-predictions"
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(tensor, tmp.name)
            s3_client.upload_file(tmp.name, bucket, s3_key)
            print(f"    ↑ Uploaded extracted latent to S3: {s3_key}")
            os.unlink(tmp.name)
    except Exception as e:
        print(f"    Failed to upload latent to S3: {e}")

def process_dates(dates, model, device, static_vars_ds, extracted_latents, download_path, s3_client):
    latents_list = []
    
    for i, day in enumerate(dates):
        print(f"  [{i+1}/{len(dates)}] Processing {day}...", flush=True)
        
        # Check S3 First
        s3_latent = fetch_latent_from_s3(s3_client, day, layer="encoder_2", hhmm="0000")
        if s3_latent is not None:
            print(f"    ✓ Downloaded latent from S3: {s3_latent.shape}")
            latents_list.append(s3_latent)
            continue
            
        print("    Latent not on S3. Running local inference pass...")
        if model is None:
            raise ValueError("Model is None but local inference is required.")
            
        download_data(day, download_path)
        # If init_hour=0, we need the previous day's data for the -6h history step
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        download_data(prev_day, download_path)
        
        batch = prepare_batch(day, download_path, init_hour=0, static_vars_ds=static_vars_ds)
        batch = batch.to(device)
        
        # Run forward pass
        for pred in rollout(model, batch, steps=1):
            pass
            
        latent = extracted_latents['latent']
        num_nans = torch.isnan(latent).sum().item()
        num_infs = torch.isinf(latent).sum().item()
        
        if num_nans > 0 or num_infs > 0:
            print(f"    WARNING: Latent contains {num_nans} NaNs and {num_infs} Infs! Skipping {day}.")
            del batch
            torch.cuda.empty_cache()
            continue
            
        latents_list.append(latent)
        print(f"    ✓ Latent captured, shape: {latent.shape}", flush=True)
        
        # Upload the extracted latent back to S3 for future runs
        upload_latent_to_s3(s3_client, latent, day)
        
        # Free memory immediately
        del batch
        torch.cuda.empty_cache()
        
    if len(latents_list) == 0:
        raise ValueError("All dates resulted in NaN/Inf latents or missing S3 files! Check input data.")
        
    return latents_list

def main():
    target_layer_name = "backbone.encoder_layers.2"  
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv"
    output_dir = Path("/scratch-shared/ekasteleyn/aurora_thesis_output/patched_rollouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_path = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s3_client = get_s3_client()
    
    # We only load the model if absolutely necessary to save memory/initialization time
    model = None
    extracted_latents = {}
    hook_handle = None
    static_vars_ds = None

    # Determine if we need to load the model (do a pre-flight check on S3)
    active_dates = load_dates(dates_csv, phase="Active")
    print(f"Running extraction for {len(active_dates)} Active dates...")
    
    needs_local_inference = False
    for day in active_dates:
        date_formatted = day.replace("-", "")
        filename = f"latent_{date_formatted}_0000_encoder_2.pt"
        s3_key = f"aurora_hres_validation/{filename}"
        if s3_client:
            try:
                s3_client.head_object(Bucket="ekasteleyn-aurora-predictions", Key=s3_key)
            except Exception:
                needs_local_inference = True
                break
        else:
            needs_local_inference = True
            break
            
    if needs_local_inference:
        print("At least one latent is missing from S3. Initializing Aurora model locally...")
        download_path.mkdir(parents=True, exist_ok=True)
        download_static(download_path)
        static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
        model = get_aurora_model()
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                extracted_latents['latent'] = output[0].detach().cpu().to(torch.float32)
            else:
                extracted_latents['latent'] = output.detach().cpu().to(torch.float32)

        target_layer = dict([*model.named_modules()])[target_layer_name]
        hook_handle = target_layer.register_forward_hook(hook_fn)

    # Extract Active Latents
    with torch.inference_mode(): 
        active_latents_list = process_dates(
            active_dates, model, device, static_vars_ds, extracted_latents, download_path, s3_client
        )
    
    if hook_handle:
        hook_handle.remove()

    # Calculate Mean Active Latent
    print("Calculating mean active latent...")
    # Stack along new dim 0 to safely calculate mean
    mean_active = torch.stack(active_latents_list).mean(dim=0)
    
    print("Saving mean active latent to disk...")
    tmp_path = output_dir / "mean_active_latent.pt.tmp"
    final_path = output_dir / "mean_active_latent.pt"
    torch.save(mean_active, tmp_path)
    os.rename(tmp_path, final_path)
    
    print(f"Done! Saved to {final_path}")

if __name__ == "__main__":
    main()
