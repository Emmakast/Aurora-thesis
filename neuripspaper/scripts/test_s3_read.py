import xarray as xr
import os
import s3fs
import warnings
warnings.filterwarnings("ignore")

storage_options = {
    "key": os.environ.get("AWS_ACCESS_KEY_ID"),
    "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL", "https://ceph-gw.science.uva.nl:8000")}
}
print("Connecting to S3...")
s3 = s3fs.S3FileSystem(**storage_options)
all_files = s3.glob("ekasteleyn-aurora-predictions/aurora_hres_validation/aurora_pred_20220101_*.nc")
print(f"Found {len(all_files)} files. Opening first 2...")

try:
    files = [s3.open(f) for f in all_files[:2]]
    ds = xr.open_mfdataset(files, engine="h5netcdf")
    print("Successfully opened dataset directly from S3!")
    print(ds.dims)
except Exception as e:
    print(f"Failed: {e}")
