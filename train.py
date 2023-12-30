#!/usr/bin/env python

import numpy
import torch
import torchvision
from torch.utils.data import DataLoader

from data_transforms import get_image_to_tensor_transform, get_tensor_to_image_transform
from forward_diffusion import add_noise_to_image

import matplotlib.pyplot as plt


BATCH_SIZE = 64


def main():

    # Get the StanfordCars dataset. As of Dec 30, 2023 the download URL is broken.
    # See https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616 for workaround.
    image_to_tensor = get_image_to_tensor_transform()
    dataset = torchvision.datasets.StanfordCars(root="data", download=True, transform=image_to_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    image_data, _ = next(iter(data_loader))
    noisy_images = add_noise_to_image(image_data[1], num_steps=30)

    show_images(noisy_images)


def show_images(images, cols=4):
    plt.figure(figsize=(15, 15))
    tensor_to_image = get_tensor_to_image_transform()

    for i, img in enumerate(images):
        plt.subplot(images.shape[0] // cols + 1, cols, i + 1)
        image = tensor_to_image(img)
        plt.imshow(image)

    plt.show()


if __name__ == "__main__":
    main()

