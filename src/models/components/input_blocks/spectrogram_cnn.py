from typing import Sequence

import torch

from src.models.components.base import (
    FlatForwardState,
    FlatInputBlock,
    SequentialModelInput,
)
from src.utils import ACTIVATIONS_MAPPING


class SpectrogramCNNEncoder(FlatInputBlock):
    """Encode log-mel spectrograms with a small 2D CNN."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        embedding_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        global_pool: str = "avgmax",
    ):
        super().__init__()

        if len(channels) == 0:
            raise ValueError("channels must contain at least one stage")

        if global_pool not in {"avg", "max", "avgmax"}:
            raise ValueError("global_pool must be one of: avg, max, avgmax")

        conv_blocks = []
        in_channels = 1
        activation_layer = ACTIVATIONS_MAPPING[activation]

        for out_channels in channels:
            conv_blocks.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if use_batch_norm:
                conv_blocks.append(torch.nn.BatchNorm2d(out_channels))
            conv_blocks.append(activation_layer())
            conv_blocks.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            conv_blocks.append(torch.nn.Dropout2d(dropout))
            in_channels = out_channels

        self.encoder = torch.nn.Sequential(*conv_blocks)
        self.global_pool = global_pool

        pooled_dim_multiplier = 2 if global_pool == "avgmax" else 1
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(channels[-1] * pooled_dim_multiplier, embedding_dim),
            activation_layer(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: SequentialModelInput) -> FlatForwardState:
        if x.numerical is None:
            raise ValueError("SpectrogramCNNEncoder expects numerical spectrogram features")

        hidden_state = x.numerical.transpose(1, 2).unsqueeze(1)
        hidden_state = self.encoder(hidden_state)

        pooled_hidden_states = []
        if self.global_pool in {"avg", "avgmax"}:
            pooled_hidden_states.append(torch.nn.functional.adaptive_avg_pool2d(hidden_state, 1).flatten(1))
        if self.global_pool in {"max", "avgmax"}:
            pooled_hidden_states.append(torch.nn.functional.adaptive_max_pool2d(hidden_state, 1).flatten(1))

        hidden_state = torch.cat(pooled_hidden_states, dim=1)
        hidden_state = self.projection(hidden_state)

        return FlatForwardState(hidden_state=hidden_state)
