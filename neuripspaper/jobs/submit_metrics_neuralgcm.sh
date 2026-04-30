#!/bin/bash
#SBATCH --job-name=eval_neuralgcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

BASE_DIR="$HOME/aurora_thesis/neuripspaper"
SCRIPT="$BASE_DIR/scripts/run_all_metrics.py"
RESULTS_DIR="$BASE_DIR/results_ifs"
LOG_DIR="$BASE_DIR/logs"
PY="$HOME/aurora_thesis/aurora_env/bin/python"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODEL="neuralgcm"
REFERENCE="ifs"
REF_ZARR="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"
YEAR="${YEAR:-2020}"
WORKERS="${WORKERS:-16}"
MODE="${MODE:-joint}"
PREDICTION_ZARR="gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr"

OUTPUT_CSV="$RESULTS_DIR/physics_evaluation_${MODEL}_${YEAR}.csv"
LOG_FILE="$LOG_DIR/physics_eval_${MODEL}_${YEAR}_$(date +%Y%m%d_%H%M%S).log"

{
    echo "========================================"
    echo "  PHYSICS EVALUATION: $MODEL"
    echo "========================================"
    echo "Job ID:       ${SLURM_JOB_ID:-local}"
    echo "Date:         $(date)"
    echo "Prediction:   $PREDICTION_ZARR"
    echo "Reference:    $REFERENCE (from $REF_ZARR)"
    echo "Output:       $OUTPUT_CSV"
    echo "========================================"
} | tee "$LOG_FILE"

eval "$PY $SCRIPT \
    --model \"$MODEL\" \
    --prediction-zarr \"$PREDICTION_ZARR\" \
    --reference \"$REFERENCE\" \
    --ref-zarr \"$REF_ZARR\" \
    --static-zarr \"gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr\" \
    --mode \"$MODE\" \
    --year \"$YEAR\" \
    --lead-times \"12h,5d,10d\" \
    --workers \"$WORKERS\" \
    --output \"$OUTPUT_CSV\"" 2>&1 | tee -a "$LOG_FILE"
