import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion(nn.Module):
    """Model which gradually adds or subtracts Gaussian noise from an image.

    Reference:
        https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model

    Variables used throughout:

        N == Num steps of this diffusion model.
        B == Batch size, or number of image samples.
        C == Number of channels in the image.  For an RGB image it'll be 3.
        H == Height of the image
        W == WidthHeight of the image
    """

    def __init__(self, num_steps):
        super().__init__()

        # Define the noise schedule (amount of noise to apply at each time step)
        betas = torch.linspace(start=0.0001, end=0.02, steps=num_steps)

        # Pre-compute noise addition co-efficients
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        # Compute noise removal co-efficients (inverse of noise addition)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recip_alphas_cumprod_minus_one = torch.sqrt(
            1.0 / alphas_cumprod - 1.0
        )

        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod_minus_one", sqrt_recip_alphas_cumprod_minus_one)

        # Pre-compute posterior mean co-efficients
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_mean_start_coeff = (
            betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )
        posterior_mean_t_coeff = (
            torch.sqrt(alphas)
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        self.register_buffer("posterior_mean_start_coeff", posterior_mean_start_coeff)
        self.register_buffer("posterior_mean_t_coeff", posterior_mean_t_coeff)

        # Pre-compute posterior variance
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    @torch.no_grad()
    def add_noise(
        self, image: torch.tensor, time_step: torch.tensor
    ) -> (torch.tensor, torch.tensor):
        """Applies noise to un-noised images (T == 0) based on a linear noise schedule at specified time steps.

        Args:
            image: image without noise of shape (B, C, H, W)
            time_step: per-image time step for noise application, of shape (B,)
        """

        # Generate noise patterns matching our image tensor (B, C, W, H)
        noise = torch.randn_like(image)

        # Extract a percentage of the source image.
        # As T increases, sqrt_alphas_cumprods decreases, so we get less and less of the original image.
        sqrt_alphas_cumprods_t = self._extract_and_reshape(
            self.sqrt_alphas_cumprod, time_step
        )
        image_component = sqrt_alphas_cumprods_t * image

        # Extract a percentage of the noise.
        # As T increases, sqrt_one_minus_alphas_cumprod increases, so we get more and more of the noise.
        sqrt_one_minus_alphas_cumprod_t = self._extract_and_reshape(
            self.sqrt_one_minus_alphas_cumprod, time_step
        )
        noise_component = sqrt_one_minus_alphas_cumprod_t * noise

        # Add the components up to produce the image(s) noised at time step T.
        image_t = image_component + noise_component

        # We also return the source "noise" applied to our images.
        return image_t, noise

    @torch.no_grad()
    def remove_noise(
        self, image_t: torch.tensor, time_step: torch.tensor, noise: torch.tensor
    ) -> torch.tensor:
        """Attempts to completely remove noise from image with noise at time step T, returning a
        prediction of the original image.

        This is basically the inverse of add_noise.

        Args:
            image_t: image with noise at timestep of shape (B, C, H, W)
            time_step: time steps associated with the image
            noise: the noise to remove
        """
        sqrt_recip_alphas_cumprod_t = self._extract_and_reshape(
            self.sqrt_recip_alphas_cumprod, time_step
        )
        scaled_image = sqrt_recip_alphas_cumprod_t * image_t

        sqrt_recip_alphas_cumprod_minus_one = self._extract_and_reshape(
            self.sqrt_recip_alphas_cumprod_minus_one, time_step
        )
        scaled_noise = sqrt_recip_alphas_cumprod_minus_one * noise

        return scaled_image - scaled_noise

    def posterior_terms(
        self, image_start: torch.tensor, image_t: torch.tensor, time_step: torch.tensor
    ) -> torch.tensor:
        """Compute the posterior mean and variance, used to decrement noise by 1 time step."""

        posterior_mean_start_coeff = self._extract_and_reshape(
            self.posterior_mean_start_coeff, time_step
        )
        posterior_mean_t_coeff = self._extract_and_reshape(
            self.posterior_mean_t_coeff, time_step
        )

        posterior_mean = (
            posterior_mean_start_coeff * image_start + posterior_mean_t_coeff * image_t
        )

        posterior_variance = self._extract_and_reshape(
            self.posterior_variance, time_step
        )
        return posterior_mean, posterior_variance

    @torch.no_grad()
    def decrement_noise(
        self, image_t: torch.tensor, time_step: torch.tensor, noise_pred: torch.tensor
    ) -> torch.tensor:
        """Removes one step of noise from an image at step T, such that we produce the image at T - 1

        Args:
            image_t: image with noise at timestep of shape (B, C, H, W)
            time_step: time steps associated with the image
            noise: the noise pattern used.
        """
        # Predict the original image by subtracting noise from it using the prediction.
        image_start = self.remove_noise(image_t, time_step, noise_pred)

        # Compute posterior mean and variance
        posterior_mean, posterior_variance = self.posterior_terms(
            image_start, image_t, time_step
        )

        sample_noise = torch.randn_like(image_t)
        return posterior_mean + torch.sqrt(posterior_variance) * sample_noise

    @staticmethod
    def _extract_and_reshape(
        data: torch.tensor, time_step: torch.tensor
    ) -> torch.tensor:
        """Utility method to extract data at specified time steps then reshape into an
        tensor which can be multiplied against the image/noise tensors.

        Args:
            data: of shape (N,)
            time_step: of shape (B,)

        Returns:
            torch.tensor: of shape (B, 1, 1, 1)
        """
        return torch.tensor([data[t] for t in time_step])[:, None, None, None].to(data.device)
