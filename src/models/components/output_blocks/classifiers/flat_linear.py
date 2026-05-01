from __future__ import annotations

from typing import Literal

import torch

from src.models.components.base import ModelContext, OutputBlock, TensorSlot
from src.utils import ACTIVATIONS_MAPPING


class FlatLinearClassifier(OutputBlock):
    def __init__(
        self,
        emb_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_classes = num_classes

        cls_layers = []
        for _ in range(n_layers - 1):
            cls_layers.append(torch.nn.Dropout(self.dropout))
            cls_layers.append(torch.nn.Linear(self.emb_dim, self.emb_dim))
            cls_layers.append(ACTIVATIONS_MAPPING[activation]())

        cls_layers.append(torch.nn.Dropout(self.dropout))
        cls_layers.append(torch.nn.Linear(self.emb_dim, num_classes))
        self.cls_layers = torch.nn.Sequential(*cls_layers)

    def forward(self, slot: TensorSlot, context: ModelContext) -> TensorSlot:
        del context
        logits = self.cls_layers(slot.value)
        return TensorSlot(value=logits, mask=slot.mask)
