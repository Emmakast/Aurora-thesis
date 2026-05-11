#!/bin/bash
#SBATCH --job-name=patch_ts_hydro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

BASE_DIR="$HOME/aurora_thesis/neuripspaper"
SCRIPT="$BASE_DIR/scripts/update_ts_hydrostatic.py"
PY="$HOME/aurora_thesis/aurora_env/bin/python"

# Model Zarrs
GC_ZARR="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
PANGU_ZARR="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
HRES_ZARR="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
NGCM_ZARR="gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr"

WORKERS=8

# Depending if you want results_ifs_hydro updated, you can change the target folder below:
TARGET_RESULTS="$BASE_DIR/results"

echo "=== Updating GraphCast Timeseries ==="
$PY $SCRIPT --csv-path "$TARGET_RESULTS/time_series_graphcast_2020.csv" --prediction-zarr "$GC_ZARR" --workers $WORKERS

echo "=== Updating Pangu Timeseries ==="
$PY $SCRIPT --csv-path "$TARGET_RESULTS/time_series_pangu_2020.csv" --prediction-zarr "$PANGU_ZARR" --workers $WORKERS

echo "=== Updating HRES Timeseries ==="
$PY $SCRIPT --csv-path "$TARGET_RESULTS/time_series_hres_2020.csv" --prediction-zarr "$HRES_ZARR" --workers $WORKERS

echo "=== Updating NeuralGCM Timeseries ==="
$PY $SCRIPT --csv-path "$TARGET_RESULTS/time_series_neuralgcm_2020.csv" --prediction-zarr "$NGCM_ZARR" --workers $WORKERS

echo "All complete!"
