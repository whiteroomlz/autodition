from __future__ import annotations

from src.models.components.base import ModelContext, OutputBlock, TensorSlot


class PassThroughOutput(OutputBlock):
    def forward(self, slot: TensorSlot, context: ModelContext) -> TensorSlot:
        del context
        return TensorSlot(value=slot.value, mask=slot.mask)
