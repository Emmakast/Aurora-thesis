#!/bin/bash
#SBATCH --job-name=sp_ablation_hypso
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

BASE_DIR="$HOME/aurora_thesis/neuripspaper"
SCRIPT="$BASE_DIR/scripts/run_all_metrics.py"
PY="$HOME/aurora_thesis/aurora_env/bin/python"

IFS_PRED_ZARR="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
OUT_FILE="$HOME/aurora_thesis/thesis/benchmark/results_ifs_hypso/physics_evaluation_ifs_ablation_hypso_2020_ifs.csv"

# Ensure the output directory exists
mkdir -p "$(dirname "$OUT_FILE")"

echo "Running Ablation: hypsometric..."
$PY $SCRIPT \
    --year 2020 \
    --model ifs_ablation_hypso \
    --prediction-zarr "$IFS_PRED_ZARR" \
    --reference ifs \
    --sp-ablation hypsometric \
    --output "$OUT_FILE"
