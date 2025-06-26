#!/bin/bash
#SBATCH --job-name=vae_cats
#SBATCH --output=vae.out
#SBATCH --error=vae.err
#SBATCH --time=03:00:00
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

module load miniconda3/3.12
module load cuda/12.6

# Initialize conda in this shell so 'conda' command and env activation work
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate mypython

# Run your training script
python vaetrain.py
