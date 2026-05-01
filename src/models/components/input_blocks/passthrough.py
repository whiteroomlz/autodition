from __future__ import annotations

from src.models.components.base import InputBlock, ModelContext, TensorSlot, ensure_single_input


class FlatNumericalPassthrough(InputBlock):
    """Pass a fixed-size tensor through unchanged."""

    def forward(self, inputs: tuple[TensorSlot, ...], context: ModelContext) -> TensorSlot:
        slot = ensure_single_input(inputs, self.__class__.__name__)
        return TensorSlot(value=slot.value, mask=slot.mask)


class SequentialNumericalPassthrough(InputBlock):
    """Pass a variable-length tensor through unchanged, preserving its mask."""

    def forward(self, inputs: tuple[TensorSlot, ...], context: ModelContext) -> TensorSlot:
        slot = ensure_single_input(inputs, self.__class__.__name__)
        return TensorSlot(value=slot.value, mask=slot.mask)
