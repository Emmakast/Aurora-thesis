#!/bin/bash
set -e

echo "Setting up symlinks..."
mkdir -p /tmp/vectors_run2
ln -sfn "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_2encoders(0,1)" /tmp/vectors_run2/AO_2encoders_0_1
ln -sfn "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_2encoders(0,2)" /tmp/vectors_run2/AO_2encoders_0_2
ln -sfn "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_3encoders" /tmp/vectors_run2/AO_3encoders

echo "Calculating indices..."
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python thesis/steering/scripts/oscillation_calculator/calculate_all_indices.py \
    --vectors-dir /tmp/vectors_run2 \
    --eof thesis/steering/scripts/oscillation_calculator/ao_loading_pattern.nc \
    --output /tmp/vectors_run2/all_ao_indices.csv

echo "Generating figures for AO_2encoders(0,1)..."
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python thesis/steering/scripts/visualization/generate_steering_figures.py \
    --phenomenon AO \
    --date 20170308 \
    --data-dir "thesis/steering/vectors/AO_2encoders(0,1)" \
    --csv-path "thesis/steering/vectors/AO_2encoders(0,1)/ao_indices.csv" \
    --name-suffix ao_ao81_polar

echo "Generating figures for AO_2encoders(0,2)..."
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python thesis/steering/scripts/visualization/generate_steering_figures.py \
    --phenomenon AO \
    --date 20170308 \
    --data-dir "thesis/steering/vectors/AO_2encoders(0,2)" \
    --csv-path "thesis/steering/vectors/AO_2encoders(0,2)/ao_indices.csv" \
    --name-suffix ao_ao81_polar

echo "Generating figures for AO_3encoders..."
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python thesis/steering/scripts/visualization/generate_steering_figures.py \
    --phenomenon AO \
    --date 20170308 \
    --data-dir "thesis/steering/vectors/AO_3encoders" \
    --csv-path "thesis/steering/vectors/AO_3encoders/ao_indices.csv" \
    --name-suffix ao_ao81_polar

echo "All complete!"
