#!/bin/bash
#SBATCH --job-name=check_tv_bug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

BASE_DIR="$HOME/aurora_thesis/neuripspaper"
PY="$HOME/aurora_thesis/aurora_env/bin/python"
SCRIPT="$BASE_DIR/scripts/check_models_tv_bug.py"

# Try these paths (adjust if needed!)
GC_ZARR="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
PANGU_ZARR="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
NGCM_ZARR="gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr"

echo "Checking GraphCast..."
$PY $SCRIPT --model GraphCast --zarr "$GC_ZARR" --csv "$BASE_DIR/results/physics_evaluation_graphcast_2020.csv"

echo "Checking Pangu..."
$PY $SCRIPT --model Pangu --zarr "$PANGU_ZARR" --csv "$BASE_DIR/results/physics_evaluation_pangu_2020.csv"

echo "Checking NeuralGCM..."
$PY $SCRIPT --model NeuralGCM --zarr "$NGCM_ZARR" --csv "$BASE_DIR/results/physics_evaluation_neuralgcm_2020.csv"
