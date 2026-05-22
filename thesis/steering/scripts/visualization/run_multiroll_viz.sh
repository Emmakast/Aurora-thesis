#!/bin/bash

MULTIROLL_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_multiroll"
SCRIPT_GLOBAL="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/visualization/visualize_alphas_grid_global.py"
SCRIPT_GRID="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/visualization/visualize_alphas_grid.py"

cd /home/ekasteleyn/aurora_thesis

echo "Scanning $MULTIROLL_DIR for step subdirectories..."

# Loop through all folders starting with steps_
for TARGET_DIR in "$MULTIROLL_DIR"/steps_*; do
    if [ ! -d "$TARGET_DIR" ]; then
        continue
    fi

    echo "========================================================"
    echo "Processing $TARGET_DIR ..."

    # Check if .png files already exist to save time
    if ls "$TARGET_DIR"/*.png 1> /dev/null 2>&1; then
        echo "Skipping $TARGET_DIR (.png files already exist)"
        continue
    fi

    # Extract the step number from the directory name
    DIR_NAME=$(basename "$TARGET_DIR")    # e.g. steps_4
    STEPS=${DIR_NAME#steps_}              # e.g. 4

    # Determine the actual date used by looking at the base file in the directory
    BASE_FILE=$(ls "$TARGET_DIR"/base_ao_*_alpha_0.0.nc 2>/dev/null | head -n 1)
    
    if [ -z "$BASE_FILE" ]; then
        echo "Warning: No base file found in $TARGET_DIR, skipping..."
        continue
    fi
    
    # Extract the date using grep/regex from the base file name (e.g., ..._20170203_1200_...)
    DATE=$(basename "$BASE_FILE" | grep -o '[0-9]\{8\}')

    # Define the exact parameters matching the multi-roll job runs
    PHENOM="AO"
    SUFFIX="ao81_steps${STEPS}"
    INIT="12"
    MASK="nomask"

    echo "Running plots for $DATE with suffix $SUFFIX"

    # Generate the global plots
    aurora_env/bin/python3 "$SCRIPT_GLOBAL" \
        --data-dir "$TARGET_DIR" \
        --phenomenon "$PHENOM" \
        --name-suffix "$SUFFIX" \
        --date "$DATE" \
        --init-hour "$INIT" \
        --mask-tag "$MASK"

    # Generate the regional plots
    aurora_env/bin/python3 "$SCRIPT_GRID" \
        --data-dir "$TARGET_DIR" \
        --phenomenon "$PHENOM" \
        --name-suffix "$SUFFIX" \
        --date "$DATE" \
        --init-hour "$INIT" \
        --mask-tag "$MASK"
done

echo "Multi-rollout visualization completed!"
