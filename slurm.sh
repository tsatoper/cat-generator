#!/bin/bash
#SBATCH --job-name=vae_cats
#SBATCH --output=vae_22.log
#SBATCH --error=vae_22.log
#SBATCH --time=10:00:00
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

module load miniconda3/3.12
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate mypython

JOB_ID=22
rm -r /hb/home/tsatoper/cat-generator/results/${JOB_ID}
python vaetrain.py \
    --batch_size 64 \
    --epochs 0 \
    --lr 1e-4 \
    --latent_dim 512 \
    --beta 4.0\
    --data_path /hb/home/tsatoper/cat-generator/data \
    --save_dir /hb/home/tsatoper/cat-generator/results/${JOB_ID} \
    --resume /hb/home/tsatoper/cat-generator/results/2/models/epoch_1770.pth \
    --job_id  ${JOB_ID}
