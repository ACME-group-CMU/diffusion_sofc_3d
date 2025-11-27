#!/bin/bash

# Base directory for configs
CONFIG_DIR="~/conditional_diffusion/configs/filtered_dataset/conditional"

# Array of config numbers
CONFIGS=({0..0})

# Submit each job
for config_num in "${CONFIGS[@]}"; do
    sbatch --job-name="train_config_${config_num}" \
           --output="./Outputs/train_config_${config_num}_%j.out" \
           --error="./Outputs/train_config_${config_num}_%j.err" \
           train.sh "$CONFIG_DIR/config_conditional$config_num.yaml"
done

echo "Submitted ${#CONFIGS[@]} training jobs"
