#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1T
#SBATCH --time=2-00:00:00
#SBATCH --output ./Outputs/%j.out
#SBATCH --error ./Outputs/%j.err
#SBATCH --mail-user=rbajpai@andrew.cmu.edu
#SBATCH --mail-type=ALL

module load aocc/4.1.0 cuda/12.4
module load anaconda/3
source activate rbenv

cd ~/conditional_diffusion/diffusion_sofc_3d

srun python3 generate_multi.py "$@"
