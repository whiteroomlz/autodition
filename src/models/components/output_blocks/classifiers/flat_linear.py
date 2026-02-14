from typing import Literal

import torch

from src.models.components.base import (
    FlatForwardState,
    ModelOutputForClassification,
    OutputBlock,
)
from src.utils import ACTIVATIONS_MAPPING


class FlatLinearClassifier(OutputBlock):
    def __init__(
        self,
        emb_dim,
        n_layers=1,
        dropout=0.0,
        activation=Literal["relu", "gelu", "tanh"],
        num_classes=1,
    ):
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

    def forward(self, x: FlatForwardState) -> ModelOutputForClassification:
        logits = self.cls_layers(x.hidden_state)

        return ModelOutputForClassification(logits=logits, meta=x.meta)
