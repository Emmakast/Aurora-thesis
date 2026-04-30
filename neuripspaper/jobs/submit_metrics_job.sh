#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# ============================================================================
# Environment & Paths
# ============================================================================
BASE_DIR="$HOME/aurora_thesis/neuripspaper"
SCRIPT="$BASE_DIR/scripts/run_all_metrics.py"
RESULTS_DIR="$BASE_DIR/results_ifs"
LOG_DIR="$BASE_DIR/logs"
PY="$HOME/aurora_thesis/aurora_env/bin/python"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ============================================================================
# Configuration Defaults (can be overridden via --export=ALL,VAR=value)
# ============================================================================
MODEL="${MODEL:-pangu}"
REFERENCE="${REFERENCE:-ifs}"
REF_ZARR="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
YEAR="${YEAR:-2020}"
WORKERS="${WORKERS:-16}"
MODE="${MODE:-joint}"

case "$MODEL" in
    pangu)
        PREDICTION_ZARR="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
        ;;
    graphcast)
        PREDICTION_ZARR="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
        STATIC_ZARR="gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
        ;;
    neuralgcm)
        PREDICTION_ZARR="gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr"
        REFERENCE="ifs"
        REF_ZARR="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"
        STATIC_ZARR="gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr"
        ;;
    fuxi)
        PREDICTION_ZARR="gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr"
        ;;
    hres)
        PREDICTION_ZARR="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
        ;;
    *)
        if [ -z "${PREDICTION_ZARR:-}" ]; then
            echo "Error: Unknown model '$MODEL' and PREDICTION_ZARR is not set." >&2
            exit 1
        fi
        ;;
esac

EXTRA_ARGS="--ref-zarr $REF_ZARR"

OUTPUT_CSV="$RESULTS_DIR/physics_evaluation_${MODEL}_${YEAR}.csv"
LOG_FILE="$LOG_DIR/physics_eval_${MODEL}_${YEAR}_$(date +%Y%m%d_%H%M%S).log"

{
    echo "========================================"
    echo "  PHYSICS EVALUATION (ALL METRICS)"
    echo "========================================"
    echo "Job ID:       ${SLURM_JOB_ID:-local}"
    echo "Date:         $(date)"
    echo "Model:        $MODEL"
    echo "Mode:         $MODE"
    echo "Reference:    $REFERENCE"
    echo "Workers:      $WORKERS"
    echo "Prediction:   $PREDICTION_ZARR"
    [ -n "$STATIC_ZARR" ] && echo "Static Zarr:  $STATIC_ZARR"
    echo "Output:       $OUTPUT_CSV"
    echo "========================================"
    echo ""
} | tee "$LOG_FILE"

# ============================================================================
# Execution
# ============================================================================
CMD="$PY $SCRIPT \
    --model \"$MODEL\" \
    --prediction-zarr \"$PREDICTION_ZARR\" \
    --reference \"$REFERENCE\" \
    --mode \"$MODE\" \
    --year \"$YEAR\" \
    --lead-times \"12h,5d,10d\" \
    --workers \"$WORKERS\" \
    --output \"$OUTPUT_CSV\""

if [ -n "$STATIC_ZARR" ]; then
    CMD="$CMD --static-zarr \"$STATIC_ZARR\""
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

eval $CMD 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

{
    echo ""
    echo "========================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  COMPLETED SUCCESSFULLY"
        echo "  Saved to: $OUTPUT_CSV"
    else
        echo "  FAILED (exit code: $EXIT_CODE)"
    fi
    echo "========================================"
    echo "Finished: $(date)"
} | tee -a "$LOG_FILE"

exit $EXIT_CODE
