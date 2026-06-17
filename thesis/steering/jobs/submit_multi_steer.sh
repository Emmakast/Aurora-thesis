#!/bin/bash

# Source vectors
SRC_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(0)"

# Target configurations
declare -A TARGETS
TARGETS["AO_2encoders(0,1)"]="1.0 1.0 0.0"
TARGETS["AO_2encoders(0,2)"]="1.0 0.0 1.0"
TARGETS["AO_3encoders"]="1.0 1.0 1.0"

for TARGET_DIR_NAME in "${!TARGETS[@]}"; do
    WEIGHTS="${TARGETS[$TARGET_DIR_NAME]}"
    TARGET_DIR="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/$TARGET_DIR_NAME"
    
    # Create directory
    mkdir -p "$TARGET_DIR"
    
    # Copy vectors with the suffix expected by steer_aurora_multi.py
    for i in 0 1 2; do
        cp "$SRC_DIR/steering_vector_ao_encoder_${i}.pt" "$TARGET_DIR/steering_vector_ao_encoder_${i}_ao81_polar.pt"
        cp "$SRC_DIR/steering_vector_norm_ao_encoder_${i}.pt" "$TARGET_DIR/steering_vector_norm_ao_encoder_${i}_ao81_polar.pt"
    done
    
    # Create job script
    JOB_FILE="/home/ekasteleyn/aurora_thesis/thesis/steering/jobs/steer_${TARGET_DIR_NAME//[()]/_}.job"
    
    cat << EOF > "$JOB_FILE"
#!/bin/bash
#SBATCH --job-name=steer_${TARGET_DIR_NAME//[()]/_}
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/steering/logs/steer_${TARGET_DIR_NAME//[()]/_}_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/steering/logs/steer_${TARGET_DIR_NAME//[()]/_}_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=192G
#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --chdir=/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/steering

module load 2023
module load PROJ/9.2.0-GCCcore-12.3.0
source /home/ekasteleyn/aurora_thesis/aurora_env/bin/activate

ALPHAS="-10.0 -5.0 -2.0 -1.0 1.0 2.0 5.0 10.0"

echo "Running steer_aurora_multi.py for $TARGET_DIR_NAME with weights: $WEIGHTS"
PYTHONUNBUFFERED=1 /home/ekasteleyn/aurora_thesis/aurora_env/bin/python steer_aurora_multi.py \\
    --phenomenon AO \\
    --alphas \$ALPHAS \\
    --layer-weights $WEIGHTS \\
    --csv /home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv \\
    --steps 12 \\
    --init-hour 12 \\
    --base-date 2017-03-08 \\
    --mask-region polar \\
    --hemisphere north \\
    --polar-lat-min 60.0 \\
    --out-dir "$TARGET_DIR" \\
    --name-suffix ao81_polar

echo "COMPLETE: \$(date)"
EOF
    
    echo "Submitting job for $TARGET_DIR_NAME"
    sbatch "$JOB_FILE"
done
