import numpy
from torchvision import transforms


def get_image_to_tensor_transform(size=(64, 64)):
    """Returns the transformation of image data into a model-compatible tensor."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales values to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1), # Scales values to [-1, 1]
    ])


def get_tensor_to_image_transform(size=(64, 64)):
    """Returns the transformation of model-compatible tensor into an image."""

    return transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2), # Scales values from [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t * 255.), # Scales value to RGB.
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # Chan, Height, Width to Height, Width, Channel
        transforms.Lambda(lambda t: t.numpy().astype(numpy.uint8)),
        transforms.ToPILImage(),
    ])
