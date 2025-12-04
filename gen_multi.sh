#!/bin/bash

# Configuration
VERSIONS=(7 8)  # Define your versions here

# Parameters
CHECKPOINT_TYPE="ALL"
NUM_SAMPLES=96  # Changed from 1292 to match your condition file size
BATCH_SIZE_PER_GPU=12
EMA_OPTIONS=("true" "false")

# Define your condition files
CONDITION_FILES=(
    "/nfs/home/6/bajpair/conditional_diffusion/datasets/generation_starters/conditions_pore_15.npy"
    "/nfs/home/6/bajpair/conditional_diffusion/datasets/generation_starters/conditions_pore_18.npy"
    "/nfs/home/6/bajpair/conditional_diffusion/datasets/generation_starters/conditions_pore_22.npy"
    "/nfs/home/6/bajpair/conditional_diffusion/datasets/generation_starters/conditions_pore_25.npy"
)

export NOISE_FILE="/nfs/home/6/bajpair/conditional_diffusion/datasets/generation_starters/noise_random.npy"

# Other parameters
export INF_TIMESTEPS=1000
export W_GUIDANCE=3.0  # You can adjust this

# Submit jobs for each version, EMA setting, AND condition file
for version in "${VERSIONS[@]}"; do
    for use_ema in "${EMA_OPTIONS[@]}"; do
        for cond_file in "${CONDITION_FILES[@]}"; do
            # Extract pore value from filename for job naming
            pore_val=$(basename "$cond_file" | sed -n 's/conditions_pore_\([0-9.]*\)\.npy/\1/p')
            
            JOB_NAME="gen_v${version}_pore${pore_val}_ema${use_ema}"
            echo "Submitting job: $JOB_NAME"
            
            # Set the condition file as environment variable
            export CONDITION_FILE="$cond_file"
            
            # Optionally customize output directory to include pore value
            export OUTPUT_DIR="./generated_samples/version_${version}_conditional_test_w_3/pore_${pore_val}_${NUM_SAMPLES}samples_ema_${use_ema}/"
            
            sbatch --job-name="$JOB_NAME" \
                   --output="./Outputs/${JOB_NAME}_%j.out" \
                   --error="./Outputs/${JOB_NAME}_%j.err" \
                   generate.sh "$version" "$CHECKPOINT_TYPE" "$NUM_SAMPLES" "$BATCH_SIZE_PER_GPU" "$use_ema"
        done
    done
done

echo "All jobs submitted! Total: $((${#VERSIONS[@]} * ${#EMA_OPTIONS[@]} * ${#CONDITION_FILES[@]})) jobs"
echo "Check with 'squeue' to monitor progress."
