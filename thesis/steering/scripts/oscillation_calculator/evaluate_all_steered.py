import os
import sys
import pandas as pd
from pathlib import Path
import xarray as xr

# Add the directory to the path so we can import the script
sys.path.append("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator")
from calculate_3_indices import calculate_indices

CLIMATOLOGY = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr"
EOFS_DIR = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices"

def extract_alpha(filename, is_base):
    import re
    alpha_match = re.search(r'alpha_(-?\d+\.?\d*)', filename)
    if alpha_match:
        return float(alpha_match.group(1))
    return 0.0 if is_base else None

def evaluate_set(name, target_index, directory):
    print(f"\n======================================")
    print(f"Evaluating {name} runs for {target_index} index")
    print(f"======================================")
    
    files = list(Path(directory).glob("*.nc"))
    
    results = []
    for f in sorted(files):
        filename = f.name
        is_base = "base_" in filename
        is_steered = "steered_" in filename
        if not is_base and not is_steered:
            continue
            
        alpha = extract_alpha(filename, is_base)
        print(f"Processing: {filename} (Alpha={alpha})")
        
        # Calculate indices using the existing function
        try:
            import logging
            logging.getLogger().setLevel(logging.ERROR)
            idx_results = calculate_indices(str(f), CLIMATOLOGY, EOFS_DIR)
            
            # The result is an xarray DataArray.
            # If it has a 'mode' dimension, select mode 0.
            arr = idx_results[target_index]
            if 'mode' in arr.dims:
                arr = arr.sel(mode=0)
            
            # If it has a 'time' dimension, select the last time step.
            if 'time' in arr.dims or arr.ndim > 0:
                val = float(arr.values[-1])
            else:
                val = float(arr.values)
                
            results.append({
                "Filename": filename,
                "Type": "Base" if is_base else "Steered",
                "Alpha": alpha,
                f"{target_index}_Index (t=last)": val
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Alpha")
        print("\nResults:")
        print(df.to_markdown(index=False))
    return df

if __name__ == "__main__":
    evaluate_set("NAO", "NAO", "/scratch-shared/ekasteleyn/nao_steered/")
    evaluate_set("PNA", "PNA", "/scratch-shared/ekasteleyn/pna_neutral_steered/")
    evaluate_set("AAO", "AAO", "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AAO_1encoder(2)")
