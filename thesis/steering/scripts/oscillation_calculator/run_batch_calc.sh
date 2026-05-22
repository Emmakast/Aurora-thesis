#!/bin/bash

# Define the base directory where your results/vectors are stored
BASE_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors"

# Paths to the scripts and references
SCRIPT_CALC="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/calculate_all_indices.py"
EOF_PATTERN="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/ao_loading_pattern.nc"
OUTPUT_CSV="${BASE_DIR}/all_ao_indices.csv"

echo "Starting batch calculation of AO indices..."

# Cleanup 2020 ablation dates so they are ignored by the script
for la_idx in 0 1 2; do
    T_DIR="${BASE_DIR}/AO_1encoder(2)_la${la_idx}"
    if [ -d "$T_DIR" ]; then
        mkdir -p "${T_DIR}/archived_2020"
        mv "${T_DIR}"/*20201202* "${T_DIR}/archived_2020/" 2>/dev/null || true
    fi
done

# Run the python script
aurora_env/bin/python3 "$SCRIPT_CALC" \
    --vectors-dir "$BASE_DIR" \
    --eof "$EOF_PATTERN" \
    --output "$OUTPUT_CSV"

echo "Calculating completed! Check the global CSV at: $OUTPUT_CSV"
