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

module purge
module load aocc/4.1.0 cuda/12.4
module load anaconda/3
source activate rbenv

cd ~/conditional_diffusion/diffusion_sofc_3d
# Check if config file is provided as argument

# Check if config file is provided as argument
if [ -n "$1" ]; then
    CONFIG_FILE="$1"
    echo "Using provided config: $CONFIG_FILE"
else
    CONFIG_FILE="$HOME/conditional_diffusion/configs/config_unconditional7.yaml"
    echo "Using default config: $CONFIG_FILE"
fi

# Expand tilde if present in the config file path
CONFIG_FILE="${CONFIG_FILE/#\~/$HOME}"

srun python3 main.py --config "$CONFIG_FILE"
