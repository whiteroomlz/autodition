from __future__ import annotations

import torch

from src.models.components.base import HiddenBlock, ModelContext, TensorSlot


class PoolingType(torch.nn.Module):
    pooling_output_multiplier: int


def last_pooling(hidden_states: torch.Tensor, padding_mask: torch.BoolTensor, dim: int = 1) -> torch.Tensor:
    seq_lengths = torch.sum(padding_mask, dim)
    return hidden_states[torch.arange(hidden_states.size(0)), seq_lengths - 1, :]


def first_pooling(
    hidden_states: torch.Tensor,
    padding_mask: torch.BoolTensor,
    dim: int = 1,
) -> torch.Tensor:
    del padding_mask, dim
    return hidden_states[:, 0, :]


def mean_pooling(hidden_states: torch.Tensor, padding_mask: torch.BoolTensor, dim: int = 1) -> torch.Tensor:
    sum_hidden_states = torch.sum(hidden_states * padding_mask[:, :, None], dim)
    sum_mask = torch.sum(padding_mask, dim, keepdim=True)
    return sum_hidden_states / sum_mask


def max_pooling(hidden_states: torch.Tensor, padding_mask: torch.BoolTensor, dim: int = 1) -> torch.Tensor:
    inverse_mask = ~padding_mask.clone()
    float_mask = inverse_mask.type_as(hidden_states).masked_fill(
        inverse_mask, torch.finfo(hidden_states.dtype).min
    )
    mask_broadcast_to_hidden_states = float_mask[:, :, None]
    masked_hidden_states = hidden_states + mask_broadcast_to_hidden_states
    return torch.max(masked_hidden_states, dim)[0]


def min_pooling(hidden_states: torch.Tensor, padding_mask: torch.BoolTensor, dim: int = 1) -> torch.Tensor:
    inverse_mask = ~padding_mask.clone()
    float_mask = inverse_mask.type_as(hidden_states).masked_fill(
        inverse_mask, torch.finfo(hidden_states.dtype).max
    )
    mask_broadcast_to_hidden_states = float_mask[:, :, None]
    masked_hidden_states = hidden_states + mask_broadcast_to_hidden_states
    return torch.min(masked_hidden_states, dim)[0]


class LastPooling(PoolingType):
    pooling_output_multiplier = 1

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        return last_pooling(data, data_mask)


class FirstPooling(PoolingType):
    pooling_output_multiplier = 1

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        return first_pooling(data, data_mask)


class MeanPooling(PoolingType):
    pooling_output_multiplier = 1

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        return mean_pooling(data, data_mask)


class MaxPooling(PoolingType):
    pooling_output_multiplier = 1

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        return max_pooling(data, data_mask)


class MaxMinPooling(PoolingType):
    pooling_output_multiplier = 2

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        max_pooling_results = max_pooling(data, data_mask)
        min_pooling_results = mean_pooling(data, data_mask)
        return torch.cat((max_pooling_results, min_pooling_results), dim=1)


class MaxMeanPooling(PoolingType):
    pooling_output_multiplier = 2

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        max_pooling_results = max_pooling(data, data_mask)
        avg_pooling_results = mean_pooling(data, data_mask)
        return torch.cat((max_pooling_results, avg_pooling_results), dim=1)


class MaxMinMeanPooling(PoolingType):
    pooling_output_multiplier = 3

    @staticmethod
    def forward(data: torch.Tensor, data_mask: torch.BoolTensor) -> torch.Tensor:
        max_pooling_results = max_pooling(data, data_mask)
        min_pooling_results = min_pooling(data, data_mask)
        avg_pooling_results = mean_pooling(data, data_mask)
        return torch.cat(
            (max_pooling_results, min_pooling_results, avg_pooling_results), dim=1
        )


class Pooling(HiddenBlock):
    def __init__(
        self,
        emb_dim: int,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        pooling_type: PoolingType | None = None,
    ) -> None:
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

    def forward(self, slot: TensorSlot, context: ModelContext) -> TensorSlot:
        del context
        if slot.mask is None:
            raise ValueError("Pooling requires an input mask")

        hidden_state = slot.value
        padding_mask = slot.mask
        pooling_results = self.pooling_func(hidden_state, padding_mask)
        pooling_results_after_normalization = self.bn(pooling_results)
        return TensorSlot(value=pooling_results_after_normalization)
