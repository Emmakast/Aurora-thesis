#!/bin/bash
set -e
source /home/ekasteleyn/aurora_thesis/aurora_env/bin/activate

echo "Plotting NAO..."
python thesis/steering/scripts/visualization/visualize_alphas_grid.py --phenomenon NAO --name-suffix "" --date 20200206 --init-hour 12 --alphas -2.0 -1.0 1.0 2.0 --mask-tag polar_north_lat30p0 --data-dir /scratch-shared/ekasteleyn/nao_steered
python thesis/steering/scripts/visualization/visualize_alphas_grid_global.py --phenomenon NAO --name-suffix "" --date 20200206 --init-hour 12 --alphas -2.0 -1.0 1.0 2.0 --mask-tag polar_north_lat30p0 --data-dir /scratch-shared/ekasteleyn/nao_steered

echo "Plotting MJO..."
python thesis/steering/scripts/visualization/visualize_alphas_grid.py --phenomenon MJO --name-suffix "" --date 20160123 --init-hour 12 --alphas -1.0 -0.5 0.5 1.0 --mask-tag tropical_lat30p0 --data-dir thesis/results
python thesis/steering/scripts/visualization/visualize_alphas_grid_global.py --phenomenon MJO --name-suffix "" --date 20160123 --init-hour 12 --alphas -1.0 -0.5 0.5 1.0 --mask-tag tropical_lat30p0 --data-dir thesis/results

echo "Plotting ENSO..."
python thesis/steering/scripts/visualization/visualize_alphas_grid.py --phenomenon ENSO --name-suffix "" --date 20170103 --init-hour 12 --alphas -1.0 -0.5 0.5 1.0 --mask-tag tropical_lat30p0 --data-dir thesis/results
python thesis/steering/scripts/visualization/visualize_alphas_grid_global.py --phenomenon ENSO --name-suffix "" --date 20170103 --init-hour 12 --alphas -1.0 -0.5 0.5 1.0 --mask-tag tropical_lat30p0 --data-dir thesis/results

echo "Done!"
