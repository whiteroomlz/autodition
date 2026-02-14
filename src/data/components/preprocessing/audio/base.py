import random
from abc import ABC, abstractmethod

import torch


class AudioPreprocessingUnit(torch.nn.Module, ABC):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor)

    @abstractmethod
    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StochasticAudioPreprocessingUnit(AudioPreprocessingUnit, ABC):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        return self._apply(tensor)


class Skip(AudioPreprocessingUnit):
    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor
