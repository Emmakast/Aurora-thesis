#!/bin/bash

# Define the base directory where your results/vectors are stored
BASE_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors"

# Define the paths to your Python scripts
SCRIPT_GLOBAL="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/visualization/visualize_alphas_grid_global.py"
SCRIPT_GRID="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/visualization/visualize_alphas_grid.py"

echo "Scanning $BASE_DIR for directories containing .nc files..."

# Enable nullglob so that if no .png files exist, the array is empty instead of containing a literal '*.png'
shopt -s nullglob

# Cleanup 2020 ablation dates (and any old plots) so visualization defaults to 2017
for la_idx in 0 1 2; do
    T_DIR="${BASE_DIR}/AO_1encoder(2)_la${la_idx}"
    if [ -d "$T_DIR" ]; then
        mkdir -p "${T_DIR}/archived_2020"
        mv "${T_DIR}"/*20201202* "${T_DIR}/archived_2020/" 2>/dev/null || true
    fi
done

# Find all unique directories containing .nc files and loop through them (ignoring archived files)
find "$BASE_DIR" -type f -name "*.nc" ! -path "*/archived_2020/*" -exec dirname {} \; | sort -u | while read -r TARGET_DIR; do
    echo "--------------------------------------------------------"
    
    # Check if .png files already exist in this directory
    png_files=("$TARGET_DIR"/*.png)
    if [ ${#png_files[@]} -gt 0 ]; then
        echo "Skipping $TARGET_DIR (.png files already exist)"
        continue
    fi
    
    echo "Processing directory: $TARGET_DIR"
    
    # Dynamically determine arguments from the netCDF filenames in the directory
    ARGS=$(python3 -c "
import glob, sys, re, os
d = sys.argv[1]
base_files = glob.glob(d + '/base_*.nc')
if not base_files: sys.exit(0)
base = os.path.basename(base_files[0])
m = re.match(r'base_(.*)_([0-9]{8})_([0-9]{4})', base)
if m:
    phenom_str, date, init = m.groups()
    phenomenon = phenom_str.split('_')[0].upper()
    if phenomenon not in ['AO', 'AAO', 'MJO', 'ENSO']:
        phenomenon = 'AO'
    name_suffix = phenom_str[len(phenomenon)+1:] if len(phenom_str) > len(phenomenon) else ''
    init_hour = int(init[:2])
    
    steered_files = glob.glob(d + '/steered_*.nc')
    mask_tag = ''
    if steered_files:
        s = os.path.basename(steered_files[0])
        m2 = re.search(init + r'_(.*)_alpha_', s)
        if m2:
            mask_tag = m2.group(1)
    
    print(f'--phenomenon {phenomenon} ' + (f'--name-suffix {name_suffix} ' if name_suffix else '--name-suffix \"\" ') + f'--date {date} --init-hour {init_hour} ' + (f'--mask-tag {mask_tag}' if mask_tag else ''))
" "$TARGET_DIR")

    if [ -n "$ARGS" ]; then
        echo "Detected arguments: $ARGS"
        
        # Run the global visualization script
        eval "aurora_env/bin/python3 \"$SCRIPT_GLOBAL\" --data-dir \"$TARGET_DIR\" $ARGS"
        
        # Run the regional visualization script
        eval "aurora_env/bin/python3 \"$SCRIPT_GRID\" --data-dir \"$TARGET_DIR\" $ARGS"
    else
        echo "Could not detect parameters for $TARGET_DIR, falling back to defaults"
        aurora_env/bin/python3 "$SCRIPT_GLOBAL" --data-dir "$TARGET_DIR"
        aurora_env/bin/python3 "$SCRIPT_GRID" --data-dir "$TARGET_DIR"
    fi
    
    echo "Finished processing $TARGET_DIR"
done

echo "Batch visualization complete!"
