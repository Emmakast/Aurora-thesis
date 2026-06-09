import argparse
from thesis.steering.scripts.oscillation_calculator.generate_mjo_eof import *
import concurrent.futures

def run_test():
    input_zarr = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    import xarray as xr
    ds = xr.open_zarr(input_zarr)
    ds = standardize_coords(ds)
    vars_dict = extract_variables(ds)
    u200 = vars_dict['u200']
    v200 = vars_dict['v200']
    
    chunk_size = 4
    n_time = 8
    
    def process_chunk(start):
        ds_local = xr.open_zarr(input_zarr)
        ds_local = standardize_coords(ds_local)
        vars_local = extract_variables(ds_local)
        u200_local = vars_local['u200']
        v200_local = vars_local['v200']
        
        end = min(start + chunk_size, n_time)
        u_chunk = u200_local.isel(time=slice(start, end)).load()
        v_chunk = v200_local.isel(time=slice(start, end)).load()
        w = VectorWind(u_chunk, v_chunk)
        vp_chunk = w.velocitypotential()
        
        vp_ds = xr.Dataset({'vp200': vp_chunk})
        vp_tropics = slice_tropics(vp_ds)['vp200']
        vp_1d = vp_tropics.mean(dim='lat').compute()
        return start, vp_1d

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_chunk, start): start for start in range(0, n_time, chunk_size)}
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result()[0])
            except Exception as e:
                print("Error:", e)
                
run_test()
