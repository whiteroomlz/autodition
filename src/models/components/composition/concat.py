from __future__ import annotations

from typing import Sequence

import torch

from src.models.components.base import ModelContext, Ref, TensorSlot, TransformStage


class ConcatFields(TransformStage):
    def __init__(self, inputs: Sequence[Ref], output_name: str, dim: int = -1) -> None:
        super().__init__(inputs=inputs, outputs=(output_name,))
        self.dim = dim

    def forward(self, context: ModelContext) -> ModelContext:
        slots = [context.resolve_slot(ref) for ref in self.inputs]
        masks = [slot.mask for slot in slots if slot.mask is not None]

        if masks and any(mask is None for mask in [slot.mask for slot in slots]):
            raise ValueError("ConcatFields expects either all masked or all unmasked inputs")

        if masks and any(not torch.equal(masks[0], mask) for mask in masks[1:]):
            raise ValueError("ConcatFields requires all input masks to be identical")

        concatenated = torch.cat([slot.value for slot in slots], dim=self.dim)
        context.write("rep", self.outputs[0], TensorSlot(value=concatenated, mask=masks[0] if masks else None))
        return context
