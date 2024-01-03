import math

import torch
import torch.nn as nn


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
