from abc import ABC, abstractmethod
from typing import Union

import torch

from src.data.components.containers import (
    CategoricalFeatureInfo,
    FeatureSchema,
)
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
        embeddings_info: CategoricalFeatureInfo,
        max_norm: Union[float, None] = None,
        scale_grad_by_freq: bool = False,
        padding_value: int = 0,
    ):
        super().__init__()

        embeddings = []
        for vocabulary_size, embedding_dim in zip(
            embeddings_info.vocabularies_size, embeddings_info.embeddings_dim
        ):
            embeddings.append(
                torch.nn.Embedding(
                    vocabulary_size,
                    embedding_dim,
                    padding_value,
                    max_norm=max_norm,
                    scale_grad_by_freq=scale_grad_by_freq,
                )
            )
        self.embeddings = torch.nn.ModuleList(embeddings)

    def forward(self, x):
        embeddings = [embedding(x[:, :, idx]) for idx, embedding in enumerate(self.embeddings)]
        embeddings = torch.cat(embeddings, axis=-1)  # noqa
        return embeddings


class Embedder(InputBlock, ABC):
    def __init__(
        self,
        embedding_dim: int,
        feature_info: FeatureSchema,
        max_norm: Union[float, None] = None,
        scale_grad_by_freq: bool = False,
        padding_value: int = 0,
        freeze: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.feature_info = feature_info

        if feature_info.numerical:
            self.numerical_dim = len(feature_info.numerical.feature_names)
        else:
            self.numerical_dim = 0

        if feature_info.categorical:
            self.categorical_dim = sum(
                embedding_dim for embedding_dim in feature_info.categorical.embeddings_dim
            )
            self.categorical_embeddings = EmbeddingLayer(
                feature_info.categorical,
                max_norm=max_norm,
                scale_grad_by_freq=scale_grad_by_freq,
                padding_value=padding_value,
            )
        else:
            self.categorical_dim = 0

        self.out_linear = torch.nn.Linear(
            self.numerical_dim + self.categorical_dim, self.final_embedding_dim
        )

        if freeze:
            self.requires_grad_(False)

    def __len__(self):
        return self.out_size

    @abstractmethod
    def forward(self, model_input: ModelInput) -> ForwardState:
        raise NotImplementedError

    def _get_hidden_state(self, model_input: ModelInput) -> torch.Tensor:
        if self.numerical_dim > 0:
            numerical = model_input.numerical
        else:
            numerical = None

        if self.categorical_dim > 0:
            categorical = self.categorical_embeddings(model_input.categorical)
        else:
            categorical = None

        if numerical is None:
            hidden_state = categorical
        elif categorical is None:
            hidden_state = numerical
        else:
            hidden_state = self.out_linear(torch.cat([numerical, categorical], dim=-1))

        return hidden_state


class FlatEmbedder(Embedder, FlatInputBlock):
    def forward(self, model_input: ModelInput) -> FlatForwardState:
        hidden_state = self._get_hidden_state(model_input)

        return FlatForwardState(hidden_state=hidden_state)


class SequentialEmbedder(Embedder, SequentialInputBlock):
    def forward(self, model_input: SequentialModelInput) -> SequentialForwardState:
        hidden_state = self._get_hidden_state(model_input)

        return SequentialForwardState(
            hidden_state=hidden_state, padding_mask=model_input.padding_mask  # noqa
        )
