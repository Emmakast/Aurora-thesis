#!/bin/bash

# Make sure we stop if any command fails
set -e

DATE="20170308"
ALPHAS="-0.5 -0.2 -0.1 -0.05 0.05 0.1 0.2 0.5"

echo "=== Plotting Regional Grids ==="

# Extreme polar vortex (variant: ao3)
echo "Plotting extreme polar vortex (AO3)..."
python visualize_alphas_grid.py --phenomenon AO --variant ao3 --date $DATE --alphas $ALPHAS

# Less extreme polar vortex (no variant)
echo "Plotting less extreme polar vortex..."
python visualize_alphas_grid.py --phenomenon AO --date $DATE --alphas $ALPHAS

echo "=== Plotting Global Grids ==="

# Extreme polar vortex (variant: ao3)
echo "Plotting extreme polar vortex (AO3) - Global..."
python visualize_alphas_grid_global.py --phenomenon AO --variant ao3 --date $DATE --alphas $ALPHAS

# Less extreme polar vortex (no variant)
echo "Plotting less extreme polar vortex - Global..."
python visualize_alphas_grid_global.py --phenomenon AO --date $DATE --alphas $ALPHAS

echo "Done!"
