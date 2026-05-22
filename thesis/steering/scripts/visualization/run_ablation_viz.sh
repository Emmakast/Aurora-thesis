#!/bin/bash

BASE_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors"
SCRIPT_GLOBAL="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/visualization/visualize_alphas_grid_global.py"
SCRIPT_GRID="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/visualization/visualize_alphas_grid.py"

cd /home/ekasteleyn/aurora_thesis

# Loop through the ablation latents
for LA_IDX in 2; do
    TARGET_DIR="${BASE_DIR}/AO_1encoder(2)_la${LA_IDX}"
    echo "========================================================"
    echo "Processing $TARGET_DIR ..."
    
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Directory not found, skipping."
        continue
    fi

    # 1. Clean up old interfering files (like the cont_232 ones)
    mkdir -p "$TARGET_DIR/archived_old"
    mv "$TARGET_DIR"/*cont_232* "$TARGET_DIR/archived_old/" 2>/dev/null || true
    mv "$TARGET_DIR"/*20201202* "$TARGET_DIR/archived_old/" 2>/dev/null || true

    # 2. Fix the capitalized "AO" in filenames so the Python script can find them
    for f in "$TARGET_DIR"/steered_AO_*.nc; do
        [ -e "$f" ] && mv "$f" "${f/steered_AO_/steered_ao_}"
    done
    for f in "$TARGET_DIR"/base_AO_*.nc; do
        [ -e "$f" ] && mv "$f" "${f/base_AO_/base_ao_}"
    done

    # 3. Define the exact parameters matching the 2017 ablation runs
    PHENOM="AO"
    SUFFIX="1encoder(2)_la${LA_IDX}"
    DATE="20170203"
    INIT="12"
    MASK="polar_both_lat60p0"

    # Fix base file missing/mismatch by pulling the known good base file from the cont10 folder
    EXPECTED_BASE="$TARGET_DIR/base_ao_${SUFFIX}_${DATE}_${INIT}00_alpha_0.0.nc"
    CONT10_BASE="${BASE_DIR}/AO_1encoder(2)_cont10/base_ao_cont_10_${DATE}_${INIT}00_alpha_0.0.nc"
    
    if [ ! -f "$EXPECTED_BASE" ]; then
        EXISTING_BASE=$(ls "$TARGET_DIR"/base_*${DATE}_${INIT}00*.nc 2>/dev/null | head -n 1)
        if [ -n "$EXISTING_BASE" ]; then
            echo "Copying local base file $EXISTING_BASE to expected name $EXPECTED_BASE"
            cp "$EXISTING_BASE" "$EXPECTED_BASE"
        elif [ -f "$CONT10_BASE" ]; then
            echo "Copying base file from cont10 folder to expected name $EXPECTED_BASE"
            cp "$CONT10_BASE" "$EXPECTED_BASE"
        else
            echo "Warning: Could not find base file for date $DATE!"
        fi
    fi

    echo "Running plots for $DATE with suffix $SUFFIX"

    # 4. Generate the plots
    aurora_env/bin/python3 "$SCRIPT_GLOBAL" \
        --data-dir "$TARGET_DIR" \
        --phenomenon "$PHENOM" \
        --name-suffix "$SUFFIX" \
        --date "$DATE" \
        --init-hour "$INIT" \
        --mask-tag "$MASK"

    aurora_env/bin/python3 "$SCRIPT_GRID" \
        --data-dir "$TARGET_DIR" \
        --phenomenon "$PHENOM" \
        --name-suffix "$SUFFIX" \
        --date "$DATE" \
        --init-hour "$INIT" \
        --mask-tag "$MASK"
done

echo "Ablation visualization completed!"
