#!/bin/bash
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output ./Outputs/%j.out
#SBATCH --error ./Outputs/%j.err
#SBATCH --mail-user=rbajpai@andrew.cmu.edu
#SBATCH --mail-type=ALL


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=0
#SBATCH --gres=gpu:1

#SBATCH --oversubscribe

module purge
module load aocc/3.2.0 cuda/11.7
module load anaconda3/2021.05
source activate rbenv

cd ~/Diffusion

export LOG_PATH="/trace/group/acmegroup/rochan/diffusion"
export CKPT_PATH="$LOG_PATH/lightning_logs/version_100162/checkpoints/last.ckpt"
export COND_PATH="/trace/group/acmegroup/rochan/final_img.npy"
#export UNCOND_PATH="/trace/home/rbajpai/group/diffusion/lightning_logs/version_57281/checkpoints/best_loss-epoch=043-loss=0.029324.ckpt"

srun python3 main.py --dir $LOG_PATH --data_path "dataset/denoised_grey_ints.npz" --ckpt $CKPT_PATH --cond_path $COND_PATH --n_epochs 100 --batch_size 8 --n_nodes 2 --img_size 96 --sample_interval 10 --data_length 10000 --lr 0.001 --dif_timesteps 1000 --inf_timesteps 1000 --base_dim 16 --n_blocks 1 --sample_size 4 --apply_sym True --n_cpu 4 --n_gpu 1 --ch_mul "(1,4,4,1)" --is_attn "(0,0,1,1)" --var_schedule "cosine" --dropout "(0,0.1,0.1,0.1)" --cond_dim 3 --train_base_model "False" --scheduler_gamma 0.6 --divide_batch True --cross_attn "False"
