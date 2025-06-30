What is the project:
Generate cat images using a Variational Autoencoder (VAE) and a Denoising Diffusion Probabilistic Model (DDPM)

What have I learned:
1. How to use a slurm to train an ML model on GPU partitions
2. How to pass and parse arguments to python scripts
3. How to use Hydra and YAML files

What have I practiced:
1. Using generative AI to write the bulk of my model, then modifying the code to run as intended
2. Testing hyperparameters and tracking results

Results: ~/cat-generator/VAE/results/(JOB_ID)/
    config.json contains hyperparameters
    models/ directory contains saved model weights
    samples.png contains 16 generated images using the most recent model from that configuration
