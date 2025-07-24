#!/bin/bash
for version in {16..16}; do
    sbatch --job-name="gen_v${version}" \
           --output="./Outputs/gen_v${version}_%j.out" \
           --error="./Outputs/gen_v${version}_%j.err" \
           generate.sh --version "${version}" \
           --num_samples 36 \
           --gpus 1 \
           --inf_timesteps 10 \
           --use_ema \
           --output_dir "./generated_samples/version_${version}" \
           --base_path "./results/lightning_logs/" \
           --pattern "*.ckpt"
done