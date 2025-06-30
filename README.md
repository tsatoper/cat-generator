What is the project:
Generate cat images using a Variational Autoencoder (VAE) and a Denoising Diffusion Probabilistic Model (DDPM)

What have I learned:
1. How to use a slurm to train an ML model on GPU partitions
2. How to pass and parse arguments to python scripts
3. How to use Hydra and YAML files
4. How to find and load datasets necessary to train a ML model

What have I practiced:
1. Using generative AI to write the bulk of my model, then modifying the code to run as intended
2. Testing hyperparameters and tracking results
3. Training VAE models

What's next:
1. Refine the hyper-paramters to the VAE loss function (especially the beta coefficient)
2. Train on labeled data sets and create a conditional generative model (Conditional Variational Autoencoder)
3. Switch from a VAE model to a DDPM model and try to recreate the results using a diffusion model

Results: ~/cat-generator/VAE/results/(JOB_ID)/
    config.json contains hyperparameters
    models/ directory contains saved model weights
    samples.png contains 16 generated images using the most recent model from that configuration
