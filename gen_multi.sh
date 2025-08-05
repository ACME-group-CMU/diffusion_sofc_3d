#!/bin/bash
for version in {2..2}; do
    sbatch --job-name="gen_v${version}" \
           --output="./Outputs/gen_v${version}_%j.out" \
           --error="./Outputs/gen_v${version}_%j.err" \
           generate.sh --version "${version}" \
           --num_samples 384 \
           --gpus 8 \
           --inf_timesteps 1000 \
           --output_dir "./generated_samples/ClippingExperiment/version_${version}" \
           --base_path "./results/lightning_logs/" \
           --pattern "*step=*.ckpt" \
           --img_size 96
done
