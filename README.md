# DiT Training Code
- Pretraining code for text-to-image latent diffusion transformer based off  [Pixart-Alpha](https://arxiv.org/pdf/2310.00426) model architecture 
- Model weights found at Jumpr/FaceBooru-DiT-256x256px
- Prompts must be structured as comma seperated Danbooru tags

## Usage
- Requires CUDA compatible device
- Requires HF and WandB accounts w/ API Keys
Install dependencies and run training
```bash
bash run.sh
```
Runs training for a model based on preconfigured parameters in config yaml file
```bash
uv run python src/train.py
```
Uploads model weights to HF
```bash
uv run python upload.py
```

## Issues
- Not HF Compatible => Full DiT class must be present for inference

## Model Architecture
- Used HF Diffusers library for DDPM scheduler and for injecting and calculating noise
- MHA self-attention over latent image patches and cross-attention for text conditioning
- adaLN-Zero conditioning for timestep conditioning
- Frozen T5 small text encoder with SiLU MLP for text prompt processing
- Classier-free guidance value of 10% training during late stage of training (last thousands of steps)
- Frozen VAE for projecting and unprojecting image pixels into latent space (REPA-E/e2e-sdvae-hf, [Repa-E Paper](https://arxiv.org/abs/2504.10483))

Reference Pixart Alpha Model Architecture (From paper)

<img width="681" height="874" alt="Pixart-Alpha model architecture" src="https://github.com/user-attachments/assets/24ce0f78-b4e2-4b48-8c1b-fb6716c0ee8e" />


| | | 
|---|---| 
| Parameters (trainable) | ~179M (excludes frozen VAE and T5-small encoder) | 
| Layers | 16 |
| Hidden size | 768 | 
| Attention heads | 12 (head dim 64) | 
| Patch size | 2 | 
| Latent channels (in/out) | 4 | 
| Resolution | 256x256 (32x32 latent, VAE downsample factor 8) | 
| Grad Clip Val | 1.0 | 
| Iterations | 250k (unsure) | 
| Max LR | 1e-4 | 
| Precision | bf16-mixed | 
| Batch Size | 96 | 
| Batch Acc | 1 | 
| Optimizer | AdamW w/ default hyperparameters | 
| LR Scheduler | OneCycleLR (10% warmup, cosine anneal) |

## Dataset
- Mix of approx 80 epochs on
     - puruchinera/anime-faces-256
     - aipracticecafe/anime-faces-256px

## Training Stack
- Pytorch
- Pytorch Lightning
- HF Diffusers and Transformers
- WandB

## Training
- GPU: RTX A6000 Pro Ampere (~45hrs)
