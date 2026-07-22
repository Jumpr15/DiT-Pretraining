# DiT Training Code
Pretraining code for text-to-image latent diffusion transformer based off  [Pixart-Alpha](https://arxiv.org/pdf/2310.00426) model architecture 
Model weights found at ...
Prompts structured as comma seperated Danbooru tags

## Usage


## Issues
- Not HF Compatible => Full DiT class must be present for inference

## Model Architecture
- Used HF Diffusers library for DDPM scheduler and for injecting and calculating noise
- MHA self-attention over latent image patches and cross-attention for text conditioning
- adaLN-Zero conditioning for timestep conditioning
- Frozen T5 small text encoder with SiLU MLP for text prompt processing
- Classier-free guidance value of 10% training during late stage of training (last thousands of steps)
- Frozen VAE for projecting and unprojecting image pixels into latent space (REPA-E/e2e-sdvae-hf, [Repa-E Paper](https://arxiv.org/abs/2504.10483))

## Dataset
- Mix of approx 80 epochs of each
     - puruchinera/anime-faces-256
     - aipracticecafe/anime-faces-256px

## Training Stack
- 

## Training
- GPU: RTX A6000 Pro Ampere (~...hrs)
