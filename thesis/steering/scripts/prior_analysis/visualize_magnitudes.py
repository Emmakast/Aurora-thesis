import os
import torch
import boto3
from dotenv import load_dotenv
import tempfile
import matplotlib.pyplot as plt
import numpy as np

# Load credentials
load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
access_key = os.getenv('UVA_S3_ACCESS_KEY')
secret_key = os.getenv('UVA_S3_SECRET_KEY')

s3_client = boto3.client(
    's3',
    endpoint_url="https://ceph-gw.science.uva.nl:8000",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

def download_latent(enc, date_str="20200102"):
    """Download a latent tensor from S3 for a specific date and encoder."""
    filename = f"latent_{date_str}_0000_{enc}.pt"
    s3_key = f"aurora_hres_validation/{filename}"
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        try:
            s3_client.download_file("ekasteleyn-aurora-predictions", s3_key, tmp.name)
            t = torch.load(tmp.name, weights_only=True, map_location='cpu').float()
            # Some dates (like 20200101) contain many NaNs, so we fall back if it's completely NaN
            if torch.isnan(t).all():
                print(f"Warning: {filename} is all NaNs.")
                os.unlink(tmp.name)
                return None
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            os.unlink(tmp.name)
            return t
        except Exception as e:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            print(f"Failed to download {filename}: {e}")
            return None

def build_polar_lat_mask(lat_size: int, lat_min: float = 60.0, hemisphere: str = "north"):
    latitudes = torch.linspace(90.0, -90.0, steps=lat_size)
    if hemisphere == "north":
        return latitudes >= lat_min
    return latitudes.abs() >= lat_min

# Directory where the steering vectors are saved
base_dir = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(0)'

shapes = {
    'encoder_0': (180, 360),
    'encoder_1': (90, 180),
    'encoder_2': (90, 180)
}

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, enc in enumerate(['encoder_0', 'encoder_1', 'encoder_2']):
    # Load steering vector
    if enc == 'encoder_0':
        sv_path = f"{base_dir}/steering_vector_ao_ao81_polar.pt"
    else:
        sv_path = f"{base_dir}/steering_vector_ao_{enc}.pt"
    sv = torch.load(sv_path, weights_only=True, map_location='cpu')
    
    # Apply polar north mask (lat >= 60.0) just like in inference
    h, w = shapes[enc]
    lat_mask = build_polar_lat_mask(h, lat_min=60.0, hemisphere="north")
    # sv shape is typically [1, seq_len, dim]
    spatial_mask = lat_mask.unsqueeze(1).expand(h, w).reshape(1, h*w, 1).to(sv.device)
    sv = sv * spatial_mask
    
    sv_norm = torch.norm(sv, dim=-1).squeeze().numpy()
    
    # Reshape for spatial plotting
    h, w = shapes[enc]
    sv_norm_spatial = sv_norm.reshape(h, w)
    
    # Plot steering vector magnitude
    ax1 = axes[i, 0]
    im1 = ax1.imshow(sv_norm_spatial, cmap='viridis')
    ax1.set_title(f'{enc} Steering Vector Mag\nMean: {sv_norm.mean():.4f}')
    plt.colorbar(im1, ax=ax1)
    
    # Download and load latent. We use 20200102 since 20200101 was mostly NaNs for enc 1 & 2.
    latent = download_latent(enc, date_str="20200102")
    if latent is None:
        latent = download_latent(enc, date_str="20180102") # fallback
    
    if latent is not None:
        latent_norm = torch.norm(latent, dim=-1).squeeze().numpy()
        latent_norm_spatial = latent_norm.reshape(h, w)
        
        # Plot latent magnitude
        ax2 = axes[i, 1]
        im2 = ax2.imshow(latent_norm_spatial, cmap='plasma')
        ax2.set_title(f'{enc} Latent Mag\nMean: {latent_norm.mean():.2f}')
        plt.colorbar(im2, ax=ax2)
        
        # Plot ratio (Steering / Latent)
        ax3 = axes[i, 2]
        valid_mask = latent_norm_spatial > 1e-6
        ratio_spatial = np.zeros_like(sv_norm_spatial)
        ratio_spatial[valid_mask] = sv_norm_spatial[valid_mask] / latent_norm_spatial[valid_mask]
        
        vmax = np.percentile(ratio_spatial[valid_mask], 99) if valid_mask.any() else 1.0
        im3 = ax3.imshow(ratio_spatial, cmap='inferno', vmax=vmax)
        ax3.set_title(f'{enc} Ratio (Steering/Latent)\nMean Ratio: {ratio_spatial[valid_mask].mean():.4f}' if valid_mask.any() else f'{enc} Ratio')
        plt.colorbar(im3, ax=ax3)
    else:
        axes[i, 1].set_title("Latent Not Found or All NaNs")
        axes[i, 2].set_title("Ratio Not Computable")

plt.suptitle("Comparison of Steering Vectors and Latent Magnitudes per Encoder (AO)", fontsize=16)
out_file = "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(0)/steering_magnitude_comparison.png"
plt.savefig(out_file, bbox_inches='tight', dpi=150)
print(f"Saved visualization to {out_file}")
