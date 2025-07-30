#!/bin/bash
for version in {14..19}; do
    sbatch --job-name="gen_v${version}" \
           --output="./Outputs/gen_v${version}_%j.out" \
           --error="./Outputs/gen_v${version}_%j.err" \
           generate.sh --version "${version}" \
           --num_samples 96 \
           --gpus 8 \
           --inf_timesteps 1000 \
           --output_dir "./generated_samples_experiments/no_ema/version_${version}" \
           --base_path "./results/lightning_logs/" \
           --pattern "*step=*.ckpt" \
           --noise_file "/nfs/home/6/bajpair/conditional_diffusion/datasets/noise_random.npy" \
           --img_size 96
done
