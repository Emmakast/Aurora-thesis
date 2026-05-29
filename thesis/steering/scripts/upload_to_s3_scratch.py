import os
import glob
import boto3
import sys

def load_env(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                os.environ[k] = v

def upload_folder(local_path, bucket_name, s3_folder):
    load_env('/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env')
    endpoint = 'https://ceph-gw.science.uva.nl:8000'
    s3_client = boto3.client('s3',
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv('UVA_S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('UVA_S3_SECRET_KEY')
    )
    
    # Get list of existing objects
    existing_keys = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    existing_keys.add(obj['Key'])
        print(f"Found {len(existing_keys)} existing objects in S3.")
    except Exception as e:
        print(f"Failed to list objects: {e}")
        
    files = glob.glob(os.path.join(local_path, "*"))
    print(f"Found {len(files)} files to upload from {local_path}")
    
    uploaded_count = 0
    skipped_count = 0
    
    for file in files:
        if os.path.isfile(file):
            s3_key = f"{s3_folder}/{os.path.basename(file)}"
            if s3_key in existing_keys:
                print(f"Skipping {file}, already exists at {s3_key}")
                skipped_count += 1
                continue
                
            print(f"Uploading {file} to {s3_key}")
            try:
                s3_client.upload_file(file, bucket_name, s3_key)
                uploaded_count += 1
            except Exception as e:
                print(f"Failed to upload {file}: {e}")
                
    print(f"Finished. Uploaded: {uploaded_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        local_path = sys.argv[1]
    else:
        # try to find it
        matches = glob.glob("/scratch-node/ekasteleyn.*/aurora_hres_latents")
        if matches:
            local_path = matches[0]
        else:
            print("Could not find scratch path")
            sys.exit(1)
            
    upload_folder(local_path, "ekasteleyn-aurora-predictions", "aurora_hres_validation")

