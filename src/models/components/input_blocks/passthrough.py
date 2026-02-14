from src.models.components.base import (
    FlatForwardState,
    FlatInputBlock,
    ModelInput,
    SequentialForwardState,
    SequentialInputBlock,
    SequentialModelInput,
)


class FlatNumericalPassthrough(FlatInputBlock):
    """Pass numerical features directly as a flat forward state (B x F)."""

    def forward(self, x: ModelInput) -> FlatForwardState:
        return FlatForwardState(hidden_state=x.numerical)


class SequentialNumericalPassthrough(SequentialInputBlock):
    """Pass numerical features directly as a sequential forward state (B x L x F).

    Expects SequentialModelInput with padding_mask.
    """

    def forward(self, x: SequentialModelInput) -> SequentialForwardState:
        return SequentialForwardState(
            hidden_state=x.numerical, padding_mask=x.padding_mask
        )
