#!/usr/bin/env python

import torch
import torchvision
import matplotlib.pyplot as plt


def main():

    # Get the StanfordCars dataset. As of Dec 30, 2023 the download URL is broken.
    # See https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616 for workaround.
    data = torchvision.datasets.StanfordCars(root="data", download=True)


    show_images(data)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def show_images(dataset, num_samples=20, cols=4):

    plt.figure(figsize=(15, 15))

    for i, img in enumerate(dataset):
        if i == num_samples:
            break

        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img[0])

    plt.show()


if __name__ == "__main__":
    main()

