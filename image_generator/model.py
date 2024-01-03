from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops.layers.torch import Rearrange

import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Used to create a sequence of embeddings to represent a position/time value, using
    sin and cosine.

    """

    def __init__(self, num_time_steps, dim):
        super().__init__()
        self._dim = dim

        position = torch.arange(num_time_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

        embedding = torch.zeros(num_time_steps, dim)
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.sin(position * div_term)

        self.register_buffer("embedding", embedding)

    def forward(self, time_step):
        time_emb = self.embedding[time_step]
        return time_emb


class UnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3):
        super().__init__()

        # Transforms time embedding into
        self._time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU(),
        )

        # First convlution
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Second convolution layer
        self._conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


    def forward(self, image, time):

        # Time vector
        t = self._time_mlp(time)
        t = F.relu(t)

        # First convolution (without time)
        x = self._conv1(image)

        # Combine image & time
        x = x + t

        # Second convolution (with time)
        x = self._conv2(x)


class Unet(nn.Module):
    """The model for predicting the noise applied to an image.

    Args:
        num_time_steps (int)
    """

    def __init__(
        self,
        num_image_channels: int = 3,
        down_sample_channels = (64, 128, 256, 512, 1024),
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
        self._conv0 = nn.Conv2d(num_image_channels, down_sample_channels[0], 3, padding=1)

        # Down sampling layers
        down_sample_blocks = [
            UnetBlock(down_sample_channels[i], down_sample_channels[i + 1], time_emb_dim)
            for i in range(len(down_sample_channels) - 1)
        ]
        self._down_sample_layers = nn.ModuleList(down_sample_blocks)

    def forward(self, image, time):

        # Create time embeddings
        time_emb = self._time_emb(time)

        # First conv layer
        x = self._conv0(image)

        # Down sampling
        residual_inputs = []
        for layer in self._down_sample_layers:
            x = layer(x, time_emb)
            residual_inputs.append(x)

        return x
