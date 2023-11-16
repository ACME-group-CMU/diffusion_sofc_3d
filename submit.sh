#!/bin/bash
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output ./Outputs/%j.out
#SBATCH --error ./Outputs/%j.err
#SBATCH --mail-user=rbajpai@andrew.cmu.edu
#SBATCH --mail-type=ALL

module purge
module load aocc/3.2.0 cuda/11.7
module load anaconda3/2021.05
source activate rbenv

cd ~/Diffusion

##export NCCL_DEBUG=INFO

srun python3 main.py --n_epochs 200 --batch_size 48  --n_nodes 12 --img_size 96 --sample_interval 20 --data_length 40000 --lr 0.0001 --dif_timesteps 1000 --inf_timesteps 50 --base_dim 16 --n_blocks 2 --sample_size 4 --apply_sym False --n_cpu 4 --mse_sum False --n_gpu 1 --is_attn "(0,0,1,1)" --ch_mul "(1,4,4,1)" --scheduler_gamma 0.985 
