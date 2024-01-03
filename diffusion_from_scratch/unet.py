from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops.layers.torch import Rearrange

import math

from .sinusoidal_pos_embeddings import SinusoidalPositionEmbeddings


class Unet(nn.Module):
    """The model for predicting the noise applied to an image.

    Args:
        num_time_steps (int)
    """

    def __init__(
        self,
        num_time_steps: int,
        time_embed_dim: int,
        time_cond_dim: int,
        num_time_tokens: int,
    ):
        super().__init__()

        # Time conditioning layers.

        self._time_hidden = nn.Sequential(
            SinusoidalPositionEmbeddings(num_time_steps, time_embed_dim),
            nn.Linear(time_embed_dim, time_cond_dim),
            nn.SiLU(),
        )

        self._to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim),
        )

        self._to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

    def forward(self, image, time_step):

        # Generate time conditioning vectors
        time_hidden = self._time_hidden(time_step)
        time_cond = self._to_time_cond(time_hidden)
        time_tokens = self._to_time_tokens(time_hidden)
        print(time_cond)
        print(time_tokens)


        return torch.tensor([])
