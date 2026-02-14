import torch

from src.models.components.base import (
    FlatForwardState,
    SeqToFlatHiddenBlock,
    SequentialForwardState,
)


class PoolingType(torch.nn.Module):
    def __init__(self):
        super().__init__()

    pooling_output_multiplier = None


def last_pooling(hidden_states, padding_mask, dim=1):
    """For right side padding."""
    seq_lengths = torch.sum(padding_mask, dim)
    last_hidden_states = hidden_states[torch.arange(hidden_states.size()[0]), seq_lengths - 1, :]
    return last_hidden_states


def first_pooling(hidden_states, padding_mask, dim=1):
    """For right side padding."""
    first_hidden_states = hidden_states[:, 0, :]
    return first_hidden_states


def mean_pooling(hidden_states, padding_mask, dim=1):
    sum_hidden_states = torch.sum(hidden_states * padding_mask[:, :, None], dim)
    sum_mask = torch.sum(padding_mask, dim, keepdim=True)
    return sum_hidden_states / sum_mask


def max_pooling(hidden_states, padding_mask, dim=1):
    inverse_mask = ~padding_mask.clone()
    float_mask = inverse_mask.type_as(hidden_states).masked_fill(
        inverse_mask, torch.finfo(hidden_states.dtype).min
    )
    mask_broadcast_to_hidden_states = float_mask[:, :, None]
    masked_hidden_states = hidden_states + mask_broadcast_to_hidden_states
    return torch.max(masked_hidden_states, dim)[0]


def min_pooling(hidden_states, padding_mask, dim=1):
    inverse_mask = ~padding_mask.clone()
    float_mask = inverse_mask.type_as(hidden_states).masked_fill(
        inverse_mask, torch.finfo(hidden_states.dtype).max
    )
    mask_broadcast_to_hidden_states = float_mask[:, :, None]
    masked_hidden_states = hidden_states + mask_broadcast_to_hidden_states
    return torch.min(masked_hidden_states, dim)[0]


class LastPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 1

    @staticmethod
    def forward(data, data_mask):
        return last_pooling(data, data_mask)


class FirstPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 1

    @staticmethod
    def forward(data, data_mask):
        return first_pooling(data, data_mask)


class MeanPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 1

    @staticmethod
    def forward(data, data_mask):
        return mean_pooling(data, data_mask)


class MaxPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 1

    @staticmethod
    def forward(data, data_mask):
        return max_pooling(data, data_mask)


class MaxMinPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 2

    @staticmethod
    def forward(data, data_mask):
        max_pooling_results = max_pooling(data, data_mask)
        min_pooling_results = mean_pooling(data, data_mask)
        pooled_data = torch.cat((max_pooling_results, min_pooling_results), dim=1)
        return pooled_data


class MaxMeanPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 2

    @staticmethod
    def forward(data, data_mask):
        max_pooling_results = max_pooling(data, data_mask)
        avg_pooling_results = mean_pooling(data, data_mask)
        pooled_data = torch.cat((max_pooling_results, avg_pooling_results), dim=1)
        return pooled_data


class MaxMinMeanPooling(PoolingType):
    def __init__(self):
        super().__init__()
        self.pooling_output_multiplier = 3

    @staticmethod
    def forward(data, data_mask):
        max_pooling_results = max_pooling(data, data_mask)
        min_pooling_results = min_pooling(data, data_mask)
        avg_pooling_results = mean_pooling(data, data_mask)
        pooled_data = torch.cat(
            (max_pooling_results, min_pooling_results, avg_pooling_results), dim=1
        )
        return pooled_data


class Pooling(SeqToFlatHiddenBlock):
    def __init__(
        self, emb_dim, use_batch_norm=False, use_layer_norm=False, pooling_type: PoolingType = None
    ):
        super().__init__()
        if use_batch_norm and use_layer_norm:
            raise ValueError("You should pass only one type of normalization")

        if pooling_type is None:
            pooling_type = MaxMinMeanPooling()

        self.pooling_func = pooling_type
        self.out_shape = pooling_type.pooling_output_multiplier * emb_dim

        if use_batch_norm:
            self.bn = torch.nn.BatchNorm1d(self.out_shape)
        elif use_layer_norm:
            self.bn = torch.nn.LayerNorm(self.out_shape)
        else:
            self.bn = torch.nn.Identity()

    def forward(self, x: SequentialForwardState) -> FlatForwardState:
        hidden_state = x.hidden_state
        padding_mask = x.padding_mask
        pooling_results = self.pooling_func(hidden_state, padding_mask)
        pooling_results_after_normalization = self.bn(pooling_results)

        return FlatForwardState(hidden_state=pooling_results_after_normalization, meta=x.meta)
