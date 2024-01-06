#!/usr/bin/env python

import os
import argparse

import numpy

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import datasets

from data_set import HuggingFaceImageDataSet
from data_transforms import get_image_to_tensor_transform
from diffusion import Diffusion
from unet import Unet
from param import HyperParameters


def main():
    parser = argparse.ArgumentParser(
        "train.py", description="Launches training for a language model"
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed value to produce deterministic results.",
        default=1337,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        help="Batch size for training.",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-io",
        "--input-optimizer-path",
        help="File path to load existing optimizer state for resuming training.",
        default="optimizer.pth",
        type=str,
    )
    parser.add_argument(
        "-ip",
        "--input-parameters-path",
        help="File path to load existing model parameters for resuming training.",
        default="parameters.pth",
        type=str,
    )
    parser.add_argument(
        "-oo",
        "--output-optimizer-path",
        help="File path to save the trained optimizer.",
        default="optimizer.pth",
        type=str,
    )
    parser.add_argument(
        "-op",
        "--output-parameters-path",
        help="File path to save the trained parameters.",
        default="parameters.pth",
        type=str,
    )

    args = parser.parse_args()

    hyper_params = HyperParameters(
        batch_size=args.batch_size,
    )
    print(f"Starting training with {hyper_params}")

    # Seed for deterministic results
    torch.manual_seed(args.seed)

    # Load the data set from Hugging faces and pass into our torch dataset wrapper.
    image_to_tensor = get_image_to_tensor_transform(hyper_params.image_size)
    hf_dataset = datasets.load_dataset("huggan/cats")["train"]
    full_dataset = HuggingFaceImageDataSet(hf_dataset, transform=image_to_tensor)

    # Split the data set into training and validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Instantiate the unet noise prediction model that we are going to train.
    unet = Unet(
        num_time_steps=hyper_params.num_time_steps,
        time_emb_dim=hyper_params.time_embed_dim,
    )
    unet.to(hyper_params.device)

    # Should we load existing parameters?
    if args.input_parameters_path:
        try:
            input_parameters = torch.load(args.input_parameters_path)
        except FileNotFoundError as e:
            print(f"Could not load {args.input_parameters_path}, skipping.")
        else:
            unet.load_state_dict(input_parameters)

    # Instantiate the optimizer.
    optimizer = torch.optim.AdamW(unet.parameters(), lr=hyper_params.learning_rate)

    # Should we load existing optimizer state?
    if args.input_optimizer_path:
        try:
            input_optimizer_state = torch.load(args.input_optimizer_path)
        except FileNotFoundError as e:
            print(f"Could not load {args.input_optimizer_path}, skipping.")
        else:
            optimizer.load_state_dict(input_optimizer_state)

    output_parameters_path = os.path.abspath(
        os.path.normpath(args.output_parameters_path)
    )
    output_optimizer_path = os.path.abspath(
        os.path.normpath(args.output_optimizer_path)
    )

    # Instantiate the model responsible for applying noise to images.
    # This model is not trained (it has no weights).
    diffusion = Diffusion(hyper_params.num_time_steps).to(hyper_params.device)

    # Instantiate our data loader instance for the training data set.
    data_loader = DataLoader(
        train_dataset, batch_size=hyper_params.batch_size, shuffle=True, drop_last=True,
    )

    # Begin training.
    for epoch in range(hyper_params.train_iters):
        for batch_index, batch in enumerate(data_loader):
            # Get input image data.
            images, _ = batch

            # Upload our image(s) to the right device.
            images = images.to(hyper_params.device)

            # Generate random noise time steps for each image.
            time_steps = torch.randint(
                0, hyper_params.num_time_steps, (hyper_params.batch_size,)
            ).long().to(hyper_params.device)

            # Apply noise to images.
            noisy_images, noises = diffusion.add_noise(images, time_steps)

            # Predict the noise that was just applied, based on the noised image & time step.
            noise_pred = unet(noisy_images, time_steps)
            loss = F.l1_loss(noises, noise_pred)

            # Zero out the gradients
            optimizer.zero_grad(set_to_none=True)

            # Propagate gradients across the computation network.
            loss.backward()

            # Update the parameters based on gradients to minimize loss.
            optimizer.step()

            if batch_index % 10 == 0:
                print(
                    f"Epoch {epoch} | batch {batch_index}/{len(data_loader)} Loss: {loss.item()} "
                )

                # Train will yield at checkpoints so we can incrementally save state.
                torch.save(unet.state_dict(), output_parameters_path)
                torch.save(optimizer.state_dict(), output_optimizer_path)

    # Train will yield at checkpoints so we can incrementally save state.
    torch.save(unet.state_dict(), output_parameters_path)
    torch.save(optimizer.state_dict(), output_optimizer_path)


if __name__ == "__main__":
    main()
