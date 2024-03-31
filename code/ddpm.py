import torch as th
import numpy as np

class DDPMSampler:

    def __init__(self, generator: th.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.120 ):
        self.betas = th.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=th.float32) ** 32
        self.alphas = 1 - self.betas
        self.alphas_cumprod = th.cumprod(self.aplhas, 0)

        self.one = th.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = th.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps / num_inference_steps
        timesteps = (np.arange(0, num_inference_steps)* step_ratio).round()[::-1].copy()
        self.timesteps = th.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int):
        prev_t = timestep - (self.num_training_steps// self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int)-> th.Tensor:
        # Computed variance using formula (7) in DDPM paper
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]

        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t

        variance = (1 - alpha_prod_prev_t) / ( 1 - alpha_prod_t) * current_beta_t

        variance = th.clamp(variance, min=1e-20)

        return variance

    def step(self, timestep: int, latents: th.Tensor, model_output: th.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1- alpha_prod_t
        beta_prod_prev_t = 1- alpha_prod_prev_t
        current_alpha_t = alpha_prod_t/ alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t

        # compute the predicted original sample using formula (15) of the DDPM paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output ) / alpha_prod_t ** 0.5

        # Compute the coefficients for the pred_orginal_sample and current sample x_t
        pred_orginal_sample_coeff = (alpha_prod_prev_t ** 0.5) * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_prev_t/ beta_prod_t 
        
        # Compute the predicted previous sample mean 
        pred_prev_sample_mean = pred_orginal_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device=model_output.device
            dtype=model_output.dtype
            noise = th.randn(model_output.shape, generator=self.generator, device=device, dtype=dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

            # N(0, 1) -> N (mu, sigma^2)
            # X = mu + sigma * Z where Z ~ N(0, 1)

            pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        

    def add_noise(self, original_samples: th.FloatTensor, timesteps: th.IntTensor)-> th.FloatTensor:

        alphas_cumprod = self.alpgas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alphas_prod = alphas_cumprod[timesteps]**0.5
        sqrt_alphas_prod = sqrt_alphas_prod.flatten()

        while len(sqrt_alphas_prod.shape) < len(original_samples.shape):
            sqrt_alphas_prod = sqrt_alphas_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # According to equation (4) of DDPM paper.
        # Z=N(0,1)
        # X = mean + stdev
        noise = th.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alphas_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise

        return noisy_samples 