from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union

import torch.nn

from src.models.components.base import (
    Block,
    FlatForwardState,
    ForwardState,
    ModelInput,
    ModelOutput,
    SequentialForwardState,
)
from src.utils.utils import recursive_merge


class Concat(Block, ABC):
    def __init__(self, blocks: Sequence[Block]):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x: Union[ModelInput | ForwardState]) -> Union[ForwardState | ModelOutput]:
        outputs = tuple(block(x) for block in self.blocks)
        outputs_concatenated = self._concat_outputs(outputs)  # noqa
        return outputs_concatenated

    @abstractmethod
    def _concat_outputs(
        self, outputs: Tuple[Union[ForwardState | ModelOutput]]
    ) -> Union[ForwardState | ModelOutput]:
        raise NotImplementedError


class ConcatFlatHidden(Concat):
    def _concat_outputs(self, outputs: Tuple[FlatForwardState]) -> FlatForwardState:
        hidden_state = torch.hstack(tuple(map(lambda output: output.hidden_state, outputs)))

        meta = dict()
        for output in outputs:
            meta = recursive_merge(meta, output.meta)

        return FlatForwardState(hidden_state=hidden_state, meta=meta)


class ConcatSeqHidden(Concat):
    def _concat_outputs(self, outputs: Tuple[SequentialForwardState]) -> SequentialForwardState:
        hidden_state = torch.hstack(tuple(map(lambda output: output.hidden_state, outputs)))
        return SequentialForwardState(
            hidden_state=hidden_state, padding_mask=outputs[0].padding_mask
        )
