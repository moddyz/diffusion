


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
            ax = plt.subplot(
                1, num_plot, (hyper_params.num_steps - step) // steps_per_img
            )
            ax.set_title(f"{step}", loc="center")
            img = tensor_to_image(image[0])
            plt.imshow(img)
            plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
