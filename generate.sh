#!/bin/bash
#SBATCH --partition=acmegroup
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output ./Outputs/%j.out
#SBATCH --error ./Outputs/%j.err
#SBATCH --mail-user=rbajpai@andrew.cmu.edu
#SBATCH --mail-type=ALL

module load aocc/4.0.0 cuda/12.8
module load anaconda3
source activate torch2

cd ~/test/conditional_diffusion/

python3 generate_multi.py "$@"