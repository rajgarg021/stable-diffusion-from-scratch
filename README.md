# Stable Diffusion from Scratch

This project implements a Stable Diffusion model from scratch using PyTorch. It provides a comprehensive, detailed approach to building and understanding each component of the Stable Diffusion architecture.

## Project Overview

This implementation covers the entire pipeline for AI image generation, including text-to-image and image-to-image capabilities. Here's a detailed breakdown of the project components and processes:

### 1. Model Architecture

The project implements several key components of the Stable Diffusion model:

a. VAE (Variational Autoencoder):
   - Encoder (VAE_Encoder): Compresses the input image into a latent space.
     - Uses convolutional layers and residual blocks for downsampling.
     - Implements the reparameterization trick: z = mean + std * noise.
     - Applies a scaling factor of 0.18215 to the latents.
   - Decoder (VAE_Decoder): Reconstructs the image from the latent space.
     - Reverses the encoding process using transposed convolutions and residual blocks.
     - Scales the input latents by 1/0.18215 before processing.

b. CLIP (Contrastive Language-Image Pre-training):
   - Implements the text encoder part of CLIP.
   - Uses a transformer architecture with 12 layers.
   - Each layer consists of self-attention, layer normalization, and feed-forward networks.
   - Includes positional embeddings added to the token embeddings.

c. UNet:
   - The core component of the diffusion process, predicting noise to be removed from the latent image.
   - Implements a U-Net architecture with skip connections.
   - Uses ResidualBlock and AttentionBlock as building blocks.
   - Incorporates cross-attention layers to condition on the text embeddings.
   - Uses GroupNorm for normalization and SiLU (Swish) as the activation function.

d. Diffusion model (LDM - Latent Diffusion Model):
   - Combines the UNet with time embeddings for the diffusion process.
   - Time embeddings are processed through a small MLP before being added to the UNet's features.

### 2. Model Loading

- Includes a custom model loader (model_loader.py) that loads pre-trained weights.
- Uses a model converter (model_converter.py) to map weights from a standard checkpoint file to the custom implementation.
- Implements a detailed mapping between the standard Stable Diffusion checkpoint and the custom implementation.
- Handles weight reshaping and concatenation where necessary (e.g., for attention layers).

### 3. Sampling Process

- The DDPMSampler class implements the sampling algorithm for the diffusion process.
- Manages the noise schedule using pre-computed alpha and beta values.
- Supports custom inference steps, allowing for faster sampling.
- Implements variance prediction for improved sample quality.
- Handles adding noise, removing noise, and managing the timesteps for the diffusion process.

### 4. Pipeline

The generate function in pipeline.py orchestrates the entire image generation process:

a. Text Encoding:
   - Input prompt is tokenized and encoded using the CLIP model.
   - For classifier-free guidance, both conditional and unconditional prompts are encoded.

b. Latent Generation:
   - For text-to-image, random latents are generated.
   - For image-to-image, the input image is encoded to latents using the VAE encoder.

c. Diffusion Process:
   - The UNet iteratively denoises the latents, guided by the text embeddings.
   - This process runs for a specified number of inference steps.

d. Image Decoding:
   - The final denoised latents are decoded into an image using the VAE decoder.

### 5. Additional Features

- Supports both text-to-image and image-to-image generation.
- Implements classifier-free guidance for better adherence to the prompt.
- Allows control over the strength of conditioning for image-to-image generation.
- Handles image normalization and denormalization.
- Converts between different image formats (PIL Image, numpy array, torch tensor).

### 6. Optimizations

- Implements memory optimizations, moving models to idle devices when not in use.
- Uses a custom SwitchSequential module to optimize memory usage during forward passes.
- Implements clamping in various parts of the code to prevent numerical instability.

### 7. Demo

- A Jupyter notebook (demo.ipynb) is provided to demonstrate the usage of the model.
- Allows setting various parameters like the prompt, sampling steps, guidance scale, etc.

## Technical Details

- Attention Mechanisms: Implements both self-attention and cross-attention using multi-head attention with customizable number of heads.
- Prompt Processing: Uses a CLIP tokenizer to convert text prompts into token IDs, handling padding and truncation.
- Classifier-Free Guidance: Implements the technique by processing both conditional and unconditional prompts, using a guidance scale to control text conditioning strength.
- Error Handling: Includes various checks and error messages for invalid inputs or configurations.
- Logging and Progress: Uses tqdm for progress bars during the sampling process.
- Customization: Allows for easy customization of various parameters like image size, number of inference steps, guidance scale, etc.

This implementation provides a deep dive into the internals of Stable Diffusion, allowing for a comprehensive understanding of the model's architecture and the diffusion process. It's a valuable resource for those looking to understand or extend state-of-the-art image generation techniques.