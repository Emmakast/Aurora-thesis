#!/bin/bash

# Define the multiroll directory containing your steps_* subfolders
MULTIROLL_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_multiroll"

# Paths to the python environment, script, and references
PYTHON_EXEC="/home/ekasteleyn/aurora_thesis/aurora_env/bin/python3"
SCRIPT_CALC="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/calculate_all_indices.py"
EOF_PATTERN="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/ao_loading_pattern.nc"
OUTPUT_CSV="${MULTIROLL_DIR}/multiroll_ao_indices.csv"

echo "Starting batch calculation of AO indices for multi-rollout..."

# By passing MULTIROLL_DIR as the vectors-dir, the python script will iterate 
# through inside folders (steps_1, steps_4, etc.) naturally.
$PYTHON_EXEC "$SCRIPT_CALC" \
    --vectors-dir "$MULTIROLL_DIR" \
    --eof "$EOF_PATTERN" \
    --output "$OUTPUT_CSV"

echo "Calculating completed! Check the multi-rollout CSV at: $OUTPUT_CSV"
