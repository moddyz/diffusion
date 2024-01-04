import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Used to create a sequence of embeddings to represent a position/time value, using
    sin and cosine."""

    def __init__(self, num_time_steps, dim):
        super().__init__()
        self._dim = dim

        # Create a sequence of the time steps in a row, then add an extra dimension to the end
        # such that we end up with [[1], [2], [3]... [N]]
        position = torch.arange(num_time_steps).unsqueeze(1)

        # Term for the position array divide against.
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

        # Alternate between sin and cos values.
        embedding = torch.zeros(num_time_steps, dim)
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("embedding", embedding)

    def forward(self, time_step):
        time_emb = self.embedding[time_step]
        return time_emb


class UnetBlock(nn.Module):
    """A building block of the Unet which down or up samples the incoming
    inputs.

    During down sampling, the width (W) & height (H) dimensions are halved, but
    the channels (C) will double.  The down sampling phase tries to extract
    the most important features of the image.

    Args:
        in_image_channels: Number of input image channels
        out_image_channels: Number of input image channels
        time_emb_dim: Size of the vector used to encode a single time step value.
        kernel_size: Kernel size for the convolution operations
        up: Whether this is an upsampling block.
    """

    def __init__(
        self,
        in_image_channels: int,
        out_image_channels: int,
        time_emb_dim: int,
        kernel_size: int = 3,
        up: bool = False,
    ):
        super().__init__()

        self._time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_image_channels),
            nn.ReLU(),
        )

        # First convolution layer
        if up:
            # During up sampling, we will concactenate input data from its symmetrical down sampling layer,
            # Hence the * 2 in input channels.
            conv = nn.Conv2d(
                in_image_channels * 2, out_image_channels, kernel_size, padding=1
            )
        else:
            conv = nn.Conv2d(
                in_image_channels, out_image_channels, kernel_size, padding=1
            )

        self._conv1 = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_image_channels),
            nn.ReLU(),
        )

        # Second convolution layer
        self._conv2 = nn.Sequential(
            nn.Conv2d(out_image_channels, out_image_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_image_channels),
            nn.ReLU(),
        )

        # Final transform
        if up:
            # The "Opposite" of Conv2d.
            self._transform = nn.ConvTranspose2d(
                out_image_channels,
                out_image_channels,
                4,
                stride=2,  # Down sample to half of original image dimensions.
                padding=1,
            )
        else:
            # This perform another convolution with a stride of 2, thus halving the original
            # image dimensions.
            self._transform = nn.Conv2d(
                out_image_channels,
                out_image_channels,
                4,  # Hmm why would the kernel size 4?
                stride=2,
                padding=1,
            )

    def forward(self, image, time):
        # Time vector
        t = self._time_mlp(time)

        # Add to extra dimensions to time so it can be added to the image tensor.
        # TODO: Is there any way to achieve this as part of the _time_mlp Sequential?
        t = t[:, :, None, None]

        # First convolution (without time)
        x = self._conv1(image)

        # Combine image & time
        x = x + t

        # Second convolution (with time)
        x = self._conv2(x)

        # Final transform
        return self._transform(x)


class Unet(nn.Module):
    """The model for predicting the noise applied to an image.

    Args:
        num_image_channels: Number of channels in the input image. Typically 3 (for R, G, B).
        down_sample_blocks: Sequence of channel sizes for the down sampling layers
        num_time_steps: Number of time steps used in the diffusion model.
        time_emb_dim: Size of the vector used to encode a single time step value.
    """

    def __init__(
        self,
        num_image_channels: int = 3,
        down_sample_channels=(64, 128, 256, 512, 1024),
        num_time_steps: int = 100,
        time_emb_dim: int = 32,
    ):
        super().__init__()

        # Time embedding layer.
        self._time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(num_time_steps, time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Initial convolution to transform our input image into dimensions compatible with the down sampling layers.
        self._conv0 = nn.Conv2d(
            num_image_channels, down_sample_channels[0], 3, padding=1
        )

        # Define the down sampling layers
        down_sample_blocks = [
            UnetBlock(
                down_sample_channels[i], down_sample_channels[i + 1], time_emb_dim
            )
            for i in range(len(down_sample_channels) - 1)
        ]
        self._down_sample_layers = nn.ModuleList(down_sample_blocks)

        # Define the up sampling layers
        up_sample_channels = list(reversed(down_sample_channels))  # Unet is symmetric
        up_sample_blocks = [
            UnetBlock(
                up_sample_channels[i], up_sample_channels[i + 1], time_emb_dim, up=True
            )
            for i in range(len(up_sample_channels) - 1)
        ]
        self._up_sample_layers = nn.ModuleList(up_sample_blocks)

        # Condense back to original image channels.
        self._output = nn.Conv2d(up_sample_channels[-1], num_image_channels, 1)

    def forward(self, image, time):
        # Create time embeddings
        time_emb = self._time_emb(time)

        # First conv layer
        x = self._conv0(image)

        # Down sampling
        residual_inputs = []
        for down in self._down_sample_layers:
            x = down(x, time_emb)
            residual_inputs.append(x)

        # Up sampling
        for up in self._up_sample_layers:
            residual_x = residual_inputs.pop()

            # Concatenate the channels.
            x = torch.cat((x, residual_x), 1)

            # Perform up sampling
            x = up(x, time_emb)

        x = self._output(x)

        return x
