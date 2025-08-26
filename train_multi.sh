#!/bin/bash

# Base directory for configs
CONFIG_DIR="~/conditional_diffusion/configs/random_seed100_clip"

# Array of config numbers
CONFIGS=({10..10})

# Submit each job
for config_num in "${CONFIGS[@]}"; do
    sbatch --job-name="train_config_${config_num}" \
           --output="./Outputs/train_config_${config_num}_%j.out" \
           --error="./Outputs/train_config_${config_num}_%j.err" \
           train.sh "$CONFIG_DIR/config_unconditional$config_num.yaml"
done

echo "Submitted ${#CONFIGS[@]} training jobs"
