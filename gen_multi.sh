#!/bin/bash

# Configuration
VERSIONS=(24 26)

# Parameters
CHECKPOINT_TYPE="step"
NUM_SAMPLES=96
BATCH_SIZE_PER_GPU=12
EMA_OPTIONS=("true")
IMG_SIZE=96

BASELINE="/nfs/home/6/bajpair/conditional_diffusion/datasets/generation_starters/"

CONDITION_FILES=("${BASELINE}conditions_Q1_f1.npy" "${BASELINE}conditions_Q2_f1.npy" "${BASELINE}conditions_Q3_f1.npy" "${BASELINE}conditions_Q4_f1.npy")

export INF_TIMESTEPS=1000
export STEP=30000
export W_GUIDANCE=4.0

for version in "${VERSIONS[@]}"; do
    for use_ema in "${EMA_OPTIONS[@]}"; do
        for condition_file in "${CONDITION_FILES[@]}"; do
            quartile=$(basename "$condition_file" .npy | cut -d'_' -f2)
            JOB_NAME="gen_v${version}_ema${use_ema}_${quartile}"
            echo "Submitting job: $JOB_NAME for quartile $quartile"
            
            sbatch --job-name="$JOB_NAME" \
       --output="./Outputs/${JOB_NAME}_%j.out" \
       --error="./Outputs/${JOB_NAME}_%j.err" \
       --export=ALL,CONDITION_FILE="$condition_file",OUTPUT_DIR="./generated_samples/filtered_dataset/version_${version}_${NUM_SAMPLES}samples_ema_${use_ema}_conditions_step${STEP}_w${W_GUIDANCE}" \
       generate.sh "$version" "$CHECKPOINT_TYPE" "$NUM_SAMPLES" "$BATCH_SIZE_PER_GPU" "$use_ema" "$IMG_SIZE" "$quartile"
        done
    done
done

echo "All jobs submitted! Check with 'squeue' to monitor progress."
