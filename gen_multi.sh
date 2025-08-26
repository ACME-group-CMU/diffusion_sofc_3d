#!/bin/bash

# Configuration
VERSIONS=(13)  # Define your versions here

# Parameters
CHECKPOINT_TYPE="ALL"
NUM_SAMPLES=96
BATCH_SIZE_PER_GPU=12
EMA_OPTIONS=("true" "false")  # Process both EMA true and false

# Optional environment variables (uncomment and set if needed)
# export CONDITION_FILE="conditions.npy"
export NOISE_FILE="/nfs/home/6/bajpair/conditional_diffusion/datasets/noise_random.npy"
export INF_TIMESTEPS=1000
# export OUTPUT_DIR="./my_custom_output"

# Submit each version with each EMA setting as separate SLURM jobs
for version in "${VERSIONS[@]}"; do
    for use_ema in "${EMA_OPTIONS[@]}"; do
        JOB_NAME="gen_v${version}_ema${use_ema}"
        echo "Submitting job: $JOB_NAME"
        
        sbatch --job-name="$JOB_NAME" \
               --output="./Outputs/${JOB_NAME}_%j.out" \
               --error="./Outputs/${JOB_NAME}_%j.err" \
               generate.sh "$version" "$CHECKPOINT_TYPE" "$NUM_SAMPLES" "$BATCH_SIZE_PER_GPU" "$use_ema"
    done
done

echo "All jobs submitted! Check with 'squeue' to monitor progress."
