from pathlib import Path
import sys
import pandas as pd
import xarray as xr
import re

sys.path.append("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator")
from calculate_all_indices import calculate_indices, standardize_coords

dirs = [
    "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_cont",
    "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_cont10",
    "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_cont232",
    "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)"
]

print("Loading climatology & EOF...")
eof_path = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc"
ds_eof = standardize_coords(xr.open_dataset(eof_path))
eof_pattern = ds_eof["eof"].squeeze()
pc1_std = float(ds_eof["pc_std"].values)

clim_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr"
ds_clim = standardize_coords(xr.open_zarr(clim_path, consolidated=True))

for d in dirs:
    d = Path(d)
    print("Processing", d)
    results = []
    for nc_file in list(d.glob("*.nc")):
        if "20170308" not in nc_file.name: continue
        is_base = "base_" in nc_file.name
        is_steered = "steered_" in nc_file.name
        if not is_base and not is_steered: continue
        
        alpha_match = re.search(r"alpha_(-?\d+\.?\d*)", nc_file.name)
        alpha = float(alpha_match.group(1)) if alpha_match else (0.0 if is_base else None)
        
        try:
            idx_legacy, idx_corrected, target_time = calculate_indices(nc_file, ds_clim, eof_pattern, pc1_std)
            results.append({
                "Filename": nc_file.name,
                "Alpha": alpha,
                "Type": "Base" if is_base else "Steered",
                "AO_Index_Legacy": idx_legacy,
                "AO_Index_Corrected": idx_corrected,
                "Target_Time": str(target_time),
                "AO": idx_corrected
            })
            print(f"Calculated {nc_file.name} alpha={alpha} -> AO={idx_corrected}")
        except Exception as e:
            print(f"Error on {nc_file.name}: {e}")
        
    if results:
        df_new = pd.DataFrame(results)
        csv_path = d / "all_indices_evaluated.csv"
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            # Remove any existing rows for 20170308 to avoid duplicates
            df_existing = df_existing[~df_existing["Filename"].str.contains("20170308")]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            print("Appended to", csv_path)
        else:
            df_new.to_csv(csv_path, index=False)
            print("Saved", csv_path)
