#!/bin/bash
#SBATCH --job-name=patch_hydro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

BASE_DIR="$HOME/aurora_thesis/neuripspaper"
SCRIPT="$BASE_DIR/scripts/update_hydrostatic.py"
PY="$HOME/aurora_thesis/aurora_env/bin/python"

# References
ERA5_ZARR="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
ERA5_LOWRES_ZARR="gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr"
IFS_ZARR="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
IFS_LOWRES_ZARR="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"

# Models (Update URLs if needed)
GC_ZARR="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
PANGU_ZARR="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
HRES_ZARR="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
NGCM_ZARR="gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr"

WORKERS=8

# echo "=== Updating GraphCast ==="
# # $PY $SCRIPT --csv-path "$BASE_DIR/results/physics_evaluation_graphcast_2020.csv" --prediction-zarr "$GC_ZARR" --ref-zarr "$ERA5_ZARR" --workers $WORKERS
# $PY $SCRIPT --csv-path "$BASE_DIR/results_ifs/physics_evaluation_graphcast_2020.csv" --prediction-zarr "$GC_ZARR" --ref-zarr "$IFS_ZARR" --workers $WORKERS

# echo "=== Updating Pangu ==="
# # $PY $SCRIPT --csv-path "$BASE_DIR/results/physics_evaluation_pangu_2020.csv" --prediction-zarr "$PANGU_ZARR" --ref-zarr "$ERA5_ZARR" --workers $WORKERS
# $PY $SCRIPT --csv-path "$BASE_DIR/results_ifs/physics_evaluation_pangu_2020.csv" --prediction-zarr "$PANGU_ZARR" --ref-zarr "$IFS_ZARR" --workers $WORKERS

# echo "=== Updating HRES ==="
# # $PY $SCRIPT --csv-path "$BASE_DIR/results/physics_evaluation_hres_2020.csv" --prediction-zarr "$HRES_ZARR" --ref-zarr "$ERA5_ZARR" --workers $WORKERS
# $PY $SCRIPT --csv-path "$BASE_DIR/results_ifs/physics_evaluation_hres_2020.csv" --prediction-zarr "$HRES_ZARR" --ref-zarr "$IFS_ZARR" --workers $WORKERS

echo "=== Updating NeuralGCM (Low Res) ==="
$PY $SCRIPT --csv-path "$BASE_DIR/results/physics_evaluation_neuralgcm_2020.csv" --prediction-zarr "$NGCM_ZARR" --ref-zarr "$ERA5_LOWRES_ZARR" --workers $WORKERS
$PY $SCRIPT --csv-path "$BASE_DIR/results_ifs/physics_evaluation_neuralgcm_2020.csv" --prediction-zarr "$NGCM_ZARR" --ref-zarr "$IFS_LOWRES_ZARR" --workers $WORKERS

echo "All complete!"
