import torch
import torch.nn.functional as F


def get_index_from_list(data, time_step):
    return torch.tensor([data[t] for t in time_step])[:, None, None, None]


class Diffusion:
    """Gradually adds or subtracts Gaussian noise from an image. """

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
    def add_noise(self, image: torch.tensor, time_step: torch.tensor):
        """Applies noise to un-noised images (T == 0) based on a linear noise schedule at specified time steps.

        Args:
            image: un-noised image (can be batched)
            time_step:
        """

        # Generate noise matching our image tensor (B, C, W, H)
        noise = torch.randn_like(image)

        # Extract a percentage of the source image.
        # As T increases, sqrt_alpha_cumprods decreases, so we get less and less of the original image.
        sqrt_alpha_cumprods = torch.tensor([self._sqrt_alphas_cumprod[t] for t in time_step])
        image_component = sqrt_alpha_cumprods[:, None, None, None] * image

        # Extract a percentage of the noise.
        # As T increases, sqrt_one_minus_alphas_cumprod increases, so we get more and more of the noise.
        sqrt_one_minus_alphas_cumprod = torch.tensor([self._sqrt_one_minus_alphas_cumprod[t] for t in time_step])
        noise_component = sqrt_one_minus_alphas_cumprod[:, None, None, None] * noise

        # Return add the components up, and provide the original noise that was used.
        return image_component + noise_component, noise

    @torch.no_grad()
    def remove_noise(self, image: torch.tensor, time_step: torch.tensor, noise: torch.tensor):
        """Removes noise from an image at time step T, such that we produce the image at T - 1,
        using specified noise.

        Args:
            image: image with noise
            time_step: the timesteps of the images
            noise: the noise to remove
        """
        # Image component
        sqrt_recip_alphas_t = get_index_from_list(self._sqrt_recip_alphas, time_step)
        image_component = sqrt_recip_alphas_t * image

        # Noise component
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self._sqrt_one_minus_alphas_cumprod, time_step
        )
        betas_t = get_index_from_list(self._betas, time_step)
        noise_component = betas_t * noise / sqrt_one_minus_alphas_cumprod_t
        model_mean = image_component - noise_component

        if time_step == 0:
            return model_mean
        else:
            noise = torch.randn_like(image)
            posterior_variance_t = get_index_from_list(self._posterior_variance, time_step)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def remove_noise_with_model(self, image, time_step, model):
        """Removes noise from an image at time step T, such that we produce the image at T - 1,
        using a Diffusion model"""
        noise_pred = model(image, time_step)
        return self.remove_with_noise(image, time_step, noise_pred)
