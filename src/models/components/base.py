from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import torch

from src.utils.setuptools import (
    SETUP_FUNCTION_NAME,
    RequiresSetupABCMeta,
    requires_setup,
)


@dataclass
class ModelInput:
    raw: Optional[Any] = None
    numerical: Optional[torch.Tensor] = None
    categorical: Optional[torch.Tensor] = None


@dataclass
class SequentialModelInput(ModelInput):
    """Sequential model input.

    B - batch size
    L - sequence length
    F - features
    """

    padding_mask: Optional[torch.Tensor] = None


@dataclass
class ForwardState(ABC):
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlatForwardState(ForwardState):
    hidden_state: torch.Tensor = None

    def __init__(self, hidden_state: torch.Tensor, meta=None):
        super().__init__(meta=meta if meta is not None else {})
        self.hidden_state = hidden_state.clone()

    @classmethod
    def clone(cls, input_forward_state):
        return cls(input_forward_state.hidden_state, meta=input_forward_state.meta)


@dataclass
class SequentialForwardState(ForwardState):
    hidden_state: torch.Tensor = None
    padding_mask: torch.BoolTensor = None

    def __init__(self, hidden_state: torch.Tensor, padding_mask: torch.BoolTensor, meta=None):
        super().__init__(meta=meta if meta is not None else {})
        self.hidden_state = hidden_state.clone()
        self.padding_mask = padding_mask.clone()

    @classmethod
    def init_without_mask(cls, hidden_state: torch.FloatTensor):
        empty_mask = hidden_state.data.new_ones(hidden_state.shape[:-1], dtype=torch.bool)
        return cls(hidden_state, empty_mask)

    @classmethod
    def clone(cls, input_forward_state):
        return cls(input_forward_state.hidden_state, input_forward_state.padding_mask)


@dataclass
class ModelOutput(ABC):
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutputForClassification(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Block(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Union[ModelInput, ForwardState]) -> Union[ForwardState, ModelOutput]:
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


def setup_modules(module: torch.nn.Module) -> None:
    setup_method = getattr(module, SETUP_FUNCTION_NAME, None)
    if callable(setup_method):
        setup_method()

    for child_module in module.children():
        setup_modules(child_module)


class Model(torch.nn.Module, ABC, metaclass=RequiresSetupABCMeta):
    @staticmethod
    def _coerce_model_input(model_input: Union[ModelInput, Dict]) -> ModelInput:
        if isinstance(model_input, dict):
            return ModelInput(**model_input)
        return model_input

    def setup(self) -> None:
        """Prepare model resources after Hydra instantiation."""

    @abstractmethod
    @requires_setup
    def forward(self, model_input: ModelInput) -> ModelOutput:
        raise NotImplementedError


class BlockModel(Model):
    def __init__(
        self,
        input_block: InputBlock,
        hidden_blocks: Sequence[HiddenBlock],
        output_block: OutputBlock,
    ):
        super().__init__()
        self.blocks = torch.nn.Sequential(input_block, *hidden_blocks, output_block)

    def forward(self, model_input: Union[ModelInput, Dict]) -> ModelOutput:
        model_input = self._coerce_model_input(model_input)
        return self.blocks(model_input)
