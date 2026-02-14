import random
from abc import ABC
from typing import Generic, Tuple, TypeVar

import torch

from .base import AudioPreprocessingUnit
from .spectrogram_augmentations import SpectrogramAugmentation
from .waveform_augmentations import WaveformAugmentation

T = TypeVar("T", WaveformAugmentation, SpectrogramAugmentation, AudioPreprocessingUnit)


class Compose(AudioPreprocessingUnit, Generic[T], ABC):
    preprocessing: Tuple[T]

    def __init__(self, preprocessing_units: Tuple[T]) -> None:
        super().__init__()

        if len(preprocessing_units) <= 1:
            raise ValueError("At least two preprocessing units are required")

        self.preprocessing_units = torch.nn.ModuleList(preprocessing_units)


class Pipeline(Compose[T]):
    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        for preprocessing in self.preprocessing_units:
            tensor = preprocessing(tensor)
        return tensor


class RandomChoice(Compose[T]):
    def __init__(self, preprocessing_units: Tuple[T]):
        super().__init__(preprocessing_units)
        self.candidates_count = len(self.preprocessing_units)

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        preprocessing_index = random.randint(0, self.candidates_count - 1)
        return self.preprocessing_units[preprocessing_index](tensor)
