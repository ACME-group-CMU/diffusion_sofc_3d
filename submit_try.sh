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

module purge
module load aocc/3.2.0 cuda/11.7
module load anaconda3/2021.05
source activate torch2

cd ~/Diffusion

srun python3 main.py --config config.yaml