#!/bin/bash

# # Ensure required libraries for reading Zarr from Google Cloud are installed
# /home/ekasteleyn/aurora_env/bin/python -m pip install --quiet fsspec gcsfs zarr

# Run the evaluation
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python /home/ekasteleyn/aurora_thesis/thesis/steering/scripts/evaluate_ao.py \
    --base /home/ekasteleyn/aurora_thesis/thesis/steering/scripts/base_ao_ao81_polar_20170308_1200_alpha_0.0.nc \
    --steered /home/ekasteleyn/aurora_thesis/thesis/steering/scripts/steered_ao_ao81_polar_20170308_1200_polar_north_lat60p0_alpha_-0.5.nc \
    --eof /home/ekasteleyn/aurora_thesis/thesis/steering/scripts/ao_loading_pattern.nc
