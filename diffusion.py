import torch
import torch.nn.functional as F


class Diffusion:
    """Adds or subtracts noise."""

    def __init__(self, num_steps):

        # Define the noise schedule (amount of noise to apply at each time step)
        self._betas = torch.linspace(start=0.0001, end=0.02, steps=num_steps)

        # This alphas is multiplied with the source image to extract a part of it.
        self._alphas = 1.0 - self._betas
        self._alphas_cumprod = torch.cumprod(self._alphas, dim=0)
        self._alphas_cumprod_prev = F.pad(self._alphas_cumprod[:-1], (1, 0), value=1.0)
        self._sqrt_recip_alphas = torch.sqrt(1.0 / self._alphas)
        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)

        # This variances component is multiplied against the noise to extract a part of that.
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self._alphas_cumprod)
        self._posterior_variance = self._betas * (1. - self._alphas_cumprod_prev) / (1. - self._alphas_cumprod)

    @torch.no_grad()
    def add_noise(self, images, time_steps):
        """Applys noise for a batch of images at specified time steps."""

        # Generate noise matching our images tensor (B, C, W, H)
        noise = torch.randn_like(images)

        # Extract a percentage of the source images.
        sqrt_alpha_cumprods = torch.tensor([self._sqrt_alphas_cumprod[t] for t in time_steps])
        image_component = sqrt_alpha_cumprods[:, None, None, None] * images

        # Extract a percentage of the noise.
        sqrt_variances = torch.tensor([self._sqrt_one_minus_alphas_cumprod[t] for t in time_steps])
        noise_component = sqrt_variances[:, None, None, None] * noise

        # Return the combination of the img & noise components, and the original noise that was used.
        return image_component + noise_component, noise

    def get_index_from_list(self, data, time_steps):
        return torch.tensor([data[t] for t in time_steps])[:, None, None, None]

    @torch.no_grad()
    def remove_noise(self, images, time_steps, model):
        """Subtract noise from a single image at a given time step, using a Diffusion model"""

        betas_t = self.get_index_from_list(self._betas, time_steps)

        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self._sqrt_one_minus_alphas_cumprod, time_steps
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self._sqrt_recip_alphas, time_steps)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            images - betas_t * model(images, time_steps) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self._posterior_variance, time_steps)

        if time_steps == 0:
            return model_mean
        else:
            noise = torch.randn_like(images)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
