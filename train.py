#!/usr/bin/env python

import numpy
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data_transforms import get_image_to_tensor_transform, get_tensor_to_image_transform
from diffusion import Diffusion
from param import HyperParameters
from model import SimpleUnet

import matplotlib.pyplot as plt


def main():

    # Define the hyper parameters.
    hyper_params = HyperParameters(
        train_iters=1,
        batch_size=10,
    )

    # Get the StanfordCars dataset. As of Dec 30, 2023 the download URL is broken.
    # See https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616 for workaround.
    image_to_tensor = get_image_to_tensor_transform(hyper_params.image_size)
    dataset = torchvision.datasets.StanfordCars(root="data", download=True, transform=image_to_tensor)
    data_loader = DataLoader(dataset, batch_size=hyper_params.batch_size, drop_last=True)

    # Instantiate the model.
    model = SimpleUnet()

    # Instantiate the model.
    diffusion = Diffusion(hyper_params.num_steps)

    # Instantiate the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)

    print(f"Starting training with {hyper_params}")

    # Begin training.
    for epoch in range(hyper_params.train_iters):

        for batch_index, batch in enumerate(data_loader):

            # Get input image data.
            images, _ = batch

            # Apply noise to images.
            time_steps = torch.randint(0, hyper_params.num_steps, (hyper_params.batch_size,)).long()
            noisy_images, noises = diffusion.add_noise(images, time_steps)

            #show_forward_diffusion(images, noisy_images, time_steps, hyper_params)

            # Compute noise prediction from noisey images.
            noise_pred = model(noisy_images, time_steps)

            # Noises pred.
            loss = F.l1_loss(noises, noise_pred)

            # Zero out the gradients
            optimizer.zero_grad(set_to_none=True)

            # Propagate gradients across the computation network.
            loss.backward()

            # Update the parameters based on gradients to minimize loss.
            optimizer.step()

            if epoch % 10 == 0 and batch_index % 10 == 0:
                print(f"Epoch {epoch} | batch {batch_index}/{len(data_loader)} Loss: {loss.item()} ")

                show_backward_diffusion(hyper_params, model, diffusion)


@torch.no_grad()
def show_backward_diffusion(hyper_params, model, diffusion):
    # Generate pure noise as a starting point.
    image = torch.randn((1, 3, *hyper_params.image_size))

    tensor_to_image = get_tensor_to_image_transform(hyper_params.image_size)

    plt.figure(figsize=(15, 15))

    model.eval()

    num_images = 10
    steps_per_img = hyper_params.num_steps // num_images

    for step in reversed(range(hyper_params.num_steps)):

        time_steps = torch.tensor(step).unsqueeze(0).long()
        image = diffusion.remove_noise(image, time_steps, model)
        image = torch.clamp(image, -1.0, 1.0)

        if step % steps_per_img == 0:
            ax = plt.subplot(1, num_images, (hyper_params.num_steps - step) // steps_per_img)
            ax.set_title(f"{step}", loc="center")
            img = tensor_to_image(image[0])
            plt.imshow(img)
            plt.axis("off")

    plt.show()
    model.train()


def show_forward_diffusion(original_images, noisy_images, time_steps, hyper_params):

    assert original_images.shape[0] == noisy_images.shape[0]

    num_images = original_images.shape[0]

    # Define a figure
    plt.figure(figsize=(15, 15))

    # Get the transform converting tensors to images.
    tensor_to_image = get_tensor_to_image_transform(hyper_params.image_size)

    # We only want to plot num_plot number of images.
    for i in range(num_images):

        ax = plt.subplot(num_images + 1, 2, i * 2 + 1)

        # Convert tensor to image.
        img = original_images[i]
        image = tensor_to_image(img)
        plt.imshow(image)
        plt.axis("off")

        # Create a subplot with noise time step as the title.
        ax = plt.subplot(num_images + 1, 2, i * 2 + 2)
        ax.set_title(f"{time_steps[i].item()}/{hyper_params.num_steps}", loc="center")

        # Convert tensor to image.
        img = noisy_images[i]
        image = tensor_to_image(img)
        plt.imshow(image)
        plt.axis("off")


    plt.show()


if __name__ == "__main__":
    main()
