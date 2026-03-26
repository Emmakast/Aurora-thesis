#!/bin/bash
#SBATCH --job-name=aurora-h100-fp64
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-h100-fp64-%j.out

echo "Starting H100 FP64 precision test..."
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Output to scratch-shared for easy access
OUTPUT_DIR="/scratch-shared/ekasteleyn/aurora_h100_fp64"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Run with 2 init times (00:00 and 12:00), 40 steps (240h rollout), FP64, save predictions
# 2 dates × 2 init hours = 4 runs, each with 40 steps = 80 predictions total
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python \
    /home/ekasteleyn/aurora_thesis/thesis/scripts/steering/extract_latents_hres.py \
    --dates "2022-01-15" "2022-01-16" \
    --init-hours 0 12 \
    --num-steps 40 \
    --fp64 \
    --save-predictions \
    --output-dir "$OUTPUT_DIR"

echo "Done!"
echo "Output files:"
ls -lh "$OUTPUT_DIR"
