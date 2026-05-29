import pandas as pd
import torch, boto3, os, io
from dotenv import load_dotenv

load_dotenv('/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env')
s3_client = boto3.client('s3', endpoint_url='https://ceph-gw.science.uva.nl:8000', aws_access_key_id=os.getenv('UVA_S3_ACCESS_KEY'), aws_secret_access_key=os.getenv('UVA_S3_SECRET_KEY'))
bucket = 'ekasteleyn-aurora-predictions'

df = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv')
active_dates = df[df['Type'] == 'Active']
latents = []
for i, row in active_dates.iterrows():
    date_str = f"{int(row['Year']):04d}{int(row['Month']):02d}{int(row['Day']):02d}"
    s3_key = f"aurora_hres_validation/latent_{date_str}_0000_encoder_2.pt"
    
    buf = io.BytesIO()
    s3_client.download_fileobj(bucket, s3_key, buf)
    buf.seek(0)
    latent = torch.load(buf, map_location='cpu')
    latents.append(latent)
    print(f"Downloaded {date_str}...")

stacked = torch.stack(latents)
print('Stacked shape:', stacked.shape)
print('NaN count before mean:', torch.isnan(stacked).sum().item())
mean_active = stacked.mean(dim=0)
print('NaN count after mean:', torch.isnan(mean_active).sum().item())
torch.save(mean_active, '/scratch-shared/ekasteleyn/aurora_thesis_output/patched_rollouts/mean_active_latent.pt')
print('Saved clean mean_active_latent.pt!')
