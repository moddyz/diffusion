#!/usr/bin/env python

"""A mini program that visualizes the forward & backward passes of a diffusion noise schedule."""

import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader

from diffusion_from_scratch.param import HyperParameters
from diffusion_from_scratch.data_transforms import (
    get_image_to_tensor_transform,
    get_tensor_to_image_transform,
)
from diffusion_from_scratch.visualize import (
    show_forward_diffusion,
    show_backward_diffusion_step,
)
from diffusion_from_scratch.diffusion import Diffusion


if __name__ == "__main__":

    torch.manual_seed(1337)

    hyper_params = HyperParameters(
        batch_size=1, # Load only a single image in a batch.
        num_steps=3,
    )

    # Noise at timestep T to apply.
    T = 2

    # Instantiate the diffusion model.
    diffusion = Diffusion(hyper_params.num_steps)

    # Load our data.
    image_to_tensor = get_image_to_tensor_transform(hyper_params.image_size)
    dataset = torchvision.datasets.StanfordCars(root="data", download=True, transform=image_to_tensor)
    data_loader = DataLoader(dataset, batch_size=hyper_params.batch_size, drop_last=True)

    # Load a single image.
    image, _ = next(iter(data_loader))

    # Apply noise to the image using the middle value of the noise schedule.
    time_step = torch.tensor((T,)).long().reshape(1, 1)

    # Note: the "noises" variable contains the raw gaussian noise that was used used to apply to the image.
    image_t, noise_t = diffusion.add_noise(image, time_step)

    # Remove noise from them.
    image_t_minus_one = diffusion.remove_noise(image_t, time_step, noise_t)
    #image_t_minus_one = torch.clamp(image_t_minus_one, -1.0, 1.0)

    # Instantiate tensor to image transform.
    tensor_to_image = get_tensor_to_image_transform(hyper_params.image_size)

    # Create the plot.
    fig = plt.figure(figsize=(15, 5))
    fig.canvas.manager.set_window_title("Illustration of Diffusion model forward & backward pass")

    # Draw original image (T == 0)
    ax = plt.subplot(1, 3, 1)
    plt.axis("off")
    ax.set_title(f"T = 0/{hyper_params.num_steps} (original image)", loc="center")
    plt.imshow(tensor_to_image(image[0]))

    # Draw the image at T
    ax = plt.subplot(1, 3, 2)
    plt.axis("off")
    ax.set_title(f"T = {time_step.item()}/{hyper_params.num_steps} (noise applied)", loc="center")
    plt.imshow(tensor_to_image(torch.clamp(image_t[0], -1.0, 1.0)))

    # Draw the image at T - 1
    ax = plt.subplot(1, 3, 3)
    plt.axis("off")
    ax.set_title(f"T = {time_step.item() - 1}/{hyper_params.num_steps} (one step of noise removed)", loc="center")
    plt.imshow(tensor_to_image(torch.clamp(image_t_minus_one[0], -1.0, 1.0)))

    plt.show()
