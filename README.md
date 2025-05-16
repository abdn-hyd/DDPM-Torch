# Denoising Diffusion Probabilistic Models

This repository implemenents the process of Denoising Diffusion Probabilistic Models (DDPM).

## Dataset

[Cifar10 Dataset(python version)](https://www.cs.toronto.edu/~kriz/cifar.html) and [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) are used in this repository.

## Train and Evaluate

Simply run the train.ipynb file to obtain the results of training and evaluation. Generated images and pth files will be stored in the outputs folder. Please check the different dataset configs in the training script.

## Results

The following GIF shows the progressive denoising from noise to clean images:

![DDPM Denoising Process](./assests/denoising_process.gif)

*Note: The animation shows the reverse diffusion process from timestep 990 to 0 (in steps of 10)*