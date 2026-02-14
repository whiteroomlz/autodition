from abc import ABC

import torch
import torchaudio

from .base import StochasticAudioPreprocessingUnit


class SpectrogramAugmentation(StochasticAudioPreprocessingUnit, ABC):
    ...


class FrequencyMasking(SpectrogramAugmentation):
    def __init__(self, freq_mask_param: int = 15, p: float = 0.5):
        super().__init__(p)
        self._mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def _apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # FrequencyMasking expects F x T, but we have T x F
        return self._mask(spectrogram.transpose(0, 1)).transpose(0, 1)


class TimeMasking(SpectrogramAugmentation):
    def __init__(self, time_mask_param: int = 40, p: float = 0.5):
        super().__init__(p)
        self._mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def _apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # TimeMasking expects F x T, but we have T x F
        return self._mask(spectrogram.transpose(0, 1)).transpose(0, 1)
