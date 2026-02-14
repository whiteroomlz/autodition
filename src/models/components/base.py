from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch


@dataclass
class ModelInput:
    raw: Optional[Tuple[Any]] = None
    numerical: Optional[torch.Tensor] = None
    categorical: Optional[torch.Tensor] = None


@dataclass
class SequentialModelInput(ModelInput):
    """Sequential model input.

        B - batch size
        L - Seq length
        F - features
    Args:
        raw: B x ...
        numerical: B x L x F
        categorical: B x L x F
        padding_mask: B x L, ignore elements with mask equals to ZERO. ( 0, False = ignore token )
    """

    padding_mask: Optional[torch.Tensor] = None


@dataclass
class ForwardState(ABC):
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlatForwardState(ForwardState):
    """Forward state of the flat block.

        B - batch size
        E - embedding size
    Args:
        hidden_state: B x E
    """

    def __init__(self, hidden_state: torch.Tensor, meta=None):
        super().__init__(meta=meta if meta is not None else dict())

        self.hidden_state = hidden_state.clone()

    @classmethod
    def clone(cls, input_forward_state):
        return cls(input_forward_state.hidden_state, meta=input_forward_state.meta)


@dataclass
class SequentialForwardState(ForwardState):
    """Forward state of the sequential block.

        B - batch size
        L - sequence length
        E - embedding size
    Args:
        hidden_state: B x L x E
        padding_mask: B x L ignore elements with mask equals to ZERO. ( 0, False = ignore token )
    """

    def __init__(self, hidden_state: torch.Tensor, padding_mask: torch.BoolTensor, meta=None):
        super().__init__(meta=meta if meta is not None else dict())

        self.hidden_state = hidden_state.clone()
        self.padding_mask = padding_mask.clone()

    @classmethod
    def init_without_mask(cls, hidden_state: torch.FloatTensor):
        empty_mask = hidden_state.data.new_ones(hidden_state.shape[:-1], dtype=torch.bool)
        return cls(hidden_state, empty_mask)  # noqa

    @classmethod
    def clone(cls, input_forward_state):
        return cls(
            input_forward_state.hidden_state,
            input_forward_state.padding_mask,
        )


@dataclass
class ModelOutput(ABC):
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutputForClassification(ModelOutput):
    """Output state of the classifier.

        B - batch size
        C - number of classes
    Args:
        logits: B x C (multiclass) or B x 1 (binary)
    """

    logits: Optional[torch.FloatTensor] = None


# region abstract


class Block(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Union[ModelInput | ForwardState]) -> Union[ForwardState | ModelOutput]:
        raise NotImplementedError


class InputBlock(Block, ABC):
    @abstractmethod
    def forward(self, x: ModelInput) -> ForwardState:
        raise NotImplementedError


class FlatInputBlock(InputBlock, ABC):
    @abstractmethod
    def forward(self, x: ModelInput) -> FlatForwardState:
        raise NotImplementedError


class SequentialInputBlock(InputBlock, ABC):
    @abstractmethod
    def forward(self, x: ModelInput) -> SequentialForwardState:
        raise NotImplementedError


class HiddenBlock(Block, ABC):
    @abstractmethod
    def forward(self, x: ForwardState) -> ForwardState:
        raise NotImplementedError


class SeqToSeqHiddenBlock(Block, ABC):
    @abstractmethod
    def forward(self, x: SequentialForwardState) -> SequentialForwardState:
        raise NotImplementedError


class SeqToFlatHiddenBlock(Block, ABC):
    @abstractmethod
    def forward(self, x: SequentialForwardState) -> FlatForwardState:
        raise NotImplementedError


class FlatToFlatHiddenBlock(Block, ABC):
    @abstractmethod
    def forward(self, x: FlatForwardState) -> FlatForwardState:
        raise NotImplementedError


class OutputBlock(Block, ABC):
    @abstractmethod
    def forward(self, x: ForwardState) -> ModelOutput:
        raise NotImplementedError


def setup_blocks(block):
    if getattr(block, "setup", None):
        block.setup()

    for sub_block in block.children():
        setup_blocks(sub_block)


# endregion


class Model(torch.nn.Module):
    def __init__(
        self,
        input_block: InputBlock,
        hidden_blocks: Sequence[HiddenBlock],
        output_block: OutputBlock,
    ):
        super().__init__()
        self.blocks = torch.nn.Sequential(input_block, *hidden_blocks, output_block)

    def forward(self, model_input: Union[ModelInput | Dict]) -> ModelOutput:
        if isinstance(model_input, dict):  # for ONNX support
            model_input = ModelInput(**model_input)

        model_output = self.blocks(model_input)
        return model_output
