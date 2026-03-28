from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from src.data.components.schema import CategoricalValueSpec, Schema
from src.models.components.base import (
    FlatForwardState,
    FlatInputBlock,
    ForwardState,
    InputBlock,
    ModelInput,
    SequentialForwardState,
    SequentialInputBlock,
    SequentialModelInput,
)


class EmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: Sequence[int],
        embedding_dim: int,
        max_norm: float | None = None,
        scale_grad_by_freq: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()

        embeddings = []
        for vocabulary_size in num_embeddings:
            embeddings.append(
                torch.nn.Embedding(
                    vocabulary_size,
                    embedding_dim,
                    padding_idx=padding_idx,
                    max_norm=max_norm,
                    scale_grad_by_freq=scale_grad_by_freq,
                )
            )
        self.embeddings = torch.nn.ModuleList(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = [embedding(x[..., idx]) for idx, embedding in enumerate(self.embeddings)]
        return torch.cat(embeddings, dim=-1)


class Embedder(InputBlock, ABC):
    def __init__(
        self,
        embedding_dim: int,
        schema: Schema,
        categorical_field_names: Sequence[str],
        max_norm: float | None = None,
        scale_grad_by_freq: bool = False,
        padding_idx: int = 0,
        freeze: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.schema = schema
        self.categorical_field_names = tuple(categorical_field_names)

        self.numerical_dim = 0
        self.categorical_specs: list[CategoricalValueSpec] = []
        for field_name in self.categorical_field_names:
            field_spec = self.schema.require_field(field_name)
            if not isinstance(field_spec.value, CategoricalValueSpec):
                raise TypeError(f"Field '{field_name}' is not categorical")
            self.categorical_specs.append(field_spec.value)

        self.categorical_dim = len(self.categorical_specs) * embedding_dim
        self.categorical_embeddings = (
            EmbeddingLayer(
                num_embeddings=[spec.cardinality for spec in self.categorical_specs],
                embedding_dim=embedding_dim,
                max_norm=max_norm,
                scale_grad_by_freq=scale_grad_by_freq,
                padding_idx=padding_idx,
            )
            if self.categorical_specs
            else None
        )

        self.out_linear = torch.nn.Identity()

        if freeze:
            self.requires_grad_(False)

    def __len__(self):
        return self.categorical_dim

    @abstractmethod
    def forward(self, model_input: ModelInput) -> ForwardState:
        raise NotImplementedError

    def _get_hidden_state(self, model_input: ModelInput) -> torch.Tensor:
        numerical = model_input.numerical

        if self.categorical_embeddings is not None:
            if model_input.categorical is None:
                raise ValueError("Categorical features are required by this embedder")
            categorical = self.categorical_embeddings(model_input.categorical)
        else:
            categorical = None

        if numerical is None:
            hidden_state = categorical
        elif categorical is None:
            hidden_state = numerical
        else:
            hidden_state = torch.cat([numerical, categorical], dim=-1)

        if hidden_state is None:
            raise ValueError("Embedder received no usable input features")

        return self.out_linear(hidden_state)


class FlatEmbedder(Embedder, FlatInputBlock):
    def forward(self, model_input: ModelInput) -> FlatForwardState:
        hidden_state = self._get_hidden_state(model_input)
        return FlatForwardState(hidden_state=hidden_state)


class SequentialEmbedder(Embedder, SequentialInputBlock):
    def forward(self, model_input: SequentialModelInput) -> SequentialForwardState:
        hidden_state = self._get_hidden_state(model_input)

        return SequentialForwardState(hidden_state=hidden_state, padding_mask=model_input.padding_mask)
