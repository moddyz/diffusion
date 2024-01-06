#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from param import HyperParameters
from diffusion import Diffusion
from unet import Unet
from data_transforms import get_tensor_to_image_transform


def main():
    parser = argparse.ArgumentParser(
        "generate.py", description="Generates an image from the model."
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed value to produce deterministic results.",
        default=1337,
        type=int,
    )
    parser.add_argument(
        "-ip",
        "--input-parameters-path",
        help="File path to load existing model parameters for resuming training.",
        default="parameters.pth",
        type=str,
    )

    args = parser.parse_args()


    # Seed for deterministic results
    torch.manual_seed(args.seed)

    hyper_params = HyperParameters()
    print(f"Generating image with params: {hyper_params}")

    # Instantiate our models for evaluation purposes.
    unet = Unet(
        num_time_steps=hyper_params.num_time_steps,
        time_emb_dim=hyper_params.time_embed_dim,
    ).to(hyper_params.device)
    unet.eval()

    if args.input_parameters_path:
        input_parameters = torch.load(args.input_parameters_path)
        unet.load_state_dict(input_parameters)

    diffusion = Diffusion(hyper_params.num_time_steps).to(hyper_params.device)
    diffusion.eval()

    # Generate pure noise as a starting point.
    image = torch.randn((1, 3, *hyper_params.image_size)).to(hyper_params.device)

    # Transform object to convert our tensor to image representation.
    tensor_to_image = get_tensor_to_image_transform(hyper_params.image_size)

    # We would like to plot the progression of pure noise into our final image.
    plt.figure(figsize=(15, 15))
    num_plot = 10
    steps_per_img = hyper_params.num_time_steps // num_plot

    # Start at the last time step and work backwards.
    for step in reversed(range(hyper_params.num_time_steps)):

        time_step = torch.tensor(step).unsqueeze(0).long().to(hyper_params.device)

        # Predict the noise pattern in the image.
        noise_pred = unet(image, time_step)

        # Decrement the noise from the image (to its T - 1 time step)
        image = diffusion.decrement_noise(image, time_step, noise_pred)

        # Clamp to visual range.
        image = torch.clamp(image, -1.0, 1.0)

        if step % steps_per_img == 0:
            # Plot the image
            ax = plt.subplot(
                1, num_plot, (hyper_params.num_time_steps - step) // steps_per_img
            )
            ax.set_title(f"T = {step}", loc="center")
            img = tensor_to_image(image[0])
            plt.imshow(img)
            plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
