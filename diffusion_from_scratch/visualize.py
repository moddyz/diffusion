import matplotlib.pyplot as plt

import torch

from .data_transforms import get_tensor_to_image_transform


@torch.no_grad()
def show_backward_diffusion_with_model(hyper_params, model, diffusion):
    # Generate pure noise as a starting point.
    image = torch.randn((1, 3, *hyper_params.image_size))

    tensor_to_image = get_tensor_to_image_transform(hyper_params.image_size)

    plt.figure(figsize=(15, 15))

    model.eval()

    num_plot = 10
    steps_per_img = hyper_params.num_steps // num_plot

    for step in reversed(range(hyper_params.num_steps)):

        time_steps = torch.tensor(step).unsqueeze(0).long()
        image = diffusion.remove_noise_with_model(image, time_steps, model)
        image = torch.clamp(image, -1.0, 1.0)

        if step % steps_per_img == 0:
            ax = plt.subplot(1, num_plot, (hyper_params.num_steps - step) // steps_per_img)
            ax.set_title(f"{step}", loc="center")
            img = tensor_to_image(image[0])
            plt.imshow(img)
            plt.axis("off")

    plt.show()
    model.train()


@torch.no_grad()
def show_backward_diffusion_step(image_t, time_step, noise, diffusion, hyper_params):
    """Visualize a single step in backward diffusion process starting with a image
    that contains noise, and the associated time step + noise pattern that was applied to it.

    The noise will be subtracted from the image and yield the image at T - 1.
    """
    tensor_to_image = get_tensor_to_image_transform(hyper_params.image_size)

    plt.figure(figsize=(15, 15))

    image_t_minus_one = diffusion.remove_noise(image_t, time_step, noise)
    image_t_minus_one = torch.clamp(image_t_minus_one, -1.0, 1.0)

    plt.axis("off")

    # Draw the image at T
    ax = plt.subplot(1, 2, 1)
    ax.set_title(f"{time_step.item()}", loc="center")
    plt.imshow(tensor_to_image(image_t[0]))

    # Draw the image at T - 1
    ax = plt.subplot(1, 2, 2)
    ax.set_title(f"{time_step.item() - 1}", loc="center")
    plt.imshow(tensor_to_image(image_t_minus_one[0]))


@torch.no_grad()
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
