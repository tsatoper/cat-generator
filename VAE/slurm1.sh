#!/bin/bash
#SBATCH --job-name=vae_cats
#SBATCH --output=vae_13.log
#SBATCH --error=vae_13.log
#SBATCH --time=10:00:00
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

module load miniconda3/3.12
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate mypython

JOB_ID=13
rm -r /hb/home/tsatoper/cat-generator/VAE/results/archive/${JOB_ID}
mv /hb/home/tsatoper/cat-generator/VAE/results/${JOB_ID} /hb/home/tsatoper/cat-generator/VAE/results/archive/
python vaetrain.py \
    --batch_size 64 \
    --epochs 2000 \
    --lr 1e-3 \
    --latent_dim 1024 \
    --beta 4.0\
    --data_path /hb/home/tsatoper/cat-generator/data \
    --save_dir /hb/home/tsatoper/cat-generator/VAE/results/${JOB_ID} \
    --resume null \
    --job_id  ${JOB_ID}
