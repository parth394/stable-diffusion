import torch as th
import numpy as np
from tqdm import tqdm 
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH//8
LATENT_HEIGHT = HEIGHT//8


def generate(
    prompt: str,
    uncond_prompt: str, # Negative Prompt or empty string
    input_image=None,
    strength=0.0, 
    do_cfg=True, 
    cfg_scale=7.5, 
    sampler_name="ddpm",
    n_inference_steps=50,
    models={}, 
    seed=None, 
    device=None, 
    idle_device=None,
    tokenizer=None
    ):
    
    with th.no_grad():

        if not ( 0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to_device(idle_device)
        else:
            to_idle: lambda x: x

        generator = th.Generator(device=device)

        if seed in None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models['clip']

        clip.to_device(device)

        # For classifier free guidance
        if do_cfg:
            # convert prompt into toknes using the tokenizer
            cond_toekns = tokenizer.batch_encode_plus([prompt], padding="max_lenght", max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = th.tensor(cond_tokens, dtype=th.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)

            # convert negative prompt into toknes using the tokenizer
            uncond_toekns = tokenizer.batch_encode_plus([uncond_prompt], padding="max_lenght", max_length=77).input_ids
            # (batch_size, seq_len)
            uncond_tokens = th.tensor(uncond_tokens, dtype=th.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)

            context = th.cat([cond_context, uncond_context])
        else:
            # convert prompt into toknes using the tokenizer
            cond_toekns = tokenizer.batch_encode_plus([prompt], padding="max_lenght", max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = th.tensor(cond_tokens, dtype=th.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            context = clip(cond_tokens)

        # offload the clip to cpu
        to_idle(clip)

        if sampler_name=="ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise NotImplementedError(f"{sampler_name} not implemented")
        
        latents_shape = ( 1, 4 ,LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
        
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channel)
            input_image_tensor = th.tensor(input_image_tensor, dtype=th.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueze(0)
            
            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = th.randn(latents_shape, generator=generator, device=device)

            # Run the image through the encode of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)

            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            # This is for text to image
            latents = sampler.add_noise(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latent_height, latent_width)
            model_input = latents

            if do_cfg:
                # (batch_size, 4 , latent_height, latent_width) -> (2 * batch_size, 4 , latent_height, latent_width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the model
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                cond_output, uncond_output = model_output.chunk(2, dim=0)
                model_ouput = cfg_scale * (cond_output - uncond_output) + uncond_output
            
            # remove the noise from the predicted latents + noise
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # (batch_size, channel, height, width) -> (batch_size, height, width, channel)
        images = images.permute(0, 2, 3, 1)

        images.to("cpu", dtype=th.int8).numpy()

        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min)/(old_max-old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # (160, )
    freqs = th.pow(10000, -th.arange(start=0, end=160, dtype=th.float32)/160)
    # (1, 160)
    x = th.tensor([timestep], dtype=th.float32)[:, None] * freqs[None]
    #  ( 1, 320)
    return th.cat([th.cos(x), th.sin(x)], dim=-1)

