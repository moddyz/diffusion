import torch


def add_noise_to_image(image, num_steps=100):
    # Define the noise schedule (amount of noise to apply at each time step.
    # Going with a simple linear one for now.
    betas = torch.linspace(start=0.0001, end=0.02, steps=num_steps)

    # Compute the cumulative product of variance at each time step.
    alphas_cumprod = torch.cumprod(1. - betas, dim=0)
    variances = 1.0 - alphas_cumprod

    # Create gaussian noise for each time step.
    noises = torch.randn((num_steps, *image.shape))

    # Apply noise to image at each time step.
    return torch.sqrt(alphas_cumprod)[:, None, None, None] * image + torch.sqrt(variances)[:, None, None, None] * noises
