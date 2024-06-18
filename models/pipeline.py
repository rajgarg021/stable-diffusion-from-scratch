import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

W = 512
H = 512
LATENTS_W = W // 8
LATENTS_H = H // 8

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embeddings(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def generate(
    prompt,
    uncond_prompt=None, # negative prompt or empty string
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # initializing a random number generator according to the specified seed
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # converting into a list of length seq_len=77
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (B, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (B, seq_len) -> (B, seq_len, Dim)
            cond_context = clip(cond_tokens)
            # converting into a list of length seq_len=77
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            # (B, seq_len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (B, seq_len) -> (B, seq_len, Dim)
            uncond_context = clip(uncond_tokens)
            # (B, seq_len, Dim) + (B, seq_len, Dim) -> (2 * B, seq_len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # converting into a list of length seq_len=77
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (B, seq_len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (B, seq_len) -> (B, seq_len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_H, LATENTS_W)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((W, H))
            # (H, W, C)
            input_image_tensor = np.array(input_image_tensor)
            # (H, W, C) -> (H, W, C)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (H, W, C) -> (H, W, C)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (H, W, C) -> (B, H, W, C)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (B, H, W, C) -> (B, C, H, W)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (B, 4, Latents_H, Latents_W)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (B, 4, Latents_H, Latents_W)
            latents = encoder(input_image_tensor, encoder_noise)

            # adding noise to the latents (the encoded input image)
            # (B, 4, Latents_H, Latents_W)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (B, 4, Latents_H, Latents_W)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embeddings(timestep).to(device)

            # (B, 4, Latents_H, Latents_W)
            model_input = latents

            if do_cfg:
                # (B, 4, Latents_H, Latents_W) -> (2 * B, 4, Latents_H, Latents_W)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (B, 4, Latents_H, Latents_W) -> (B, 4, Latents_H, Latents_W)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (B, 4, Latents_H, Latents_W) -> (B, 4, Latents_H, Latents_W)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (B, 4, Latents_H, Latents_W) -> (B, 3, H, W)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (B, C, H, W) -> (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    