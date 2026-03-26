import xarray as xr

# Make sure to use the 'gs://' prefix, not 'https://'
url = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"

# Use open_dataset instead of open_zarr
ds = xr.open_dataset(url, engine='zarr')

# Print the dataset to inspect the float points (e.g., float32, float64)
print(ds)