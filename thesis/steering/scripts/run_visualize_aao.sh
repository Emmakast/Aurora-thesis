#!/bin/bash

# Replace 20201202 with the date of your AAO results
DATE_TAG="20201202"

# Run the regional/polar visualization
python visualize_alphas_grid.py --phenomenon AAO --date $DATE_TAG

# Run the global visualization
python visualize_alphas_grid_global.py --phenomenon AAO --date $DATE_TAG
