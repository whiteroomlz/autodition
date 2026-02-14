import random
from abc import ABC

import torch

from .base import StochasticAudioPreprocessingUnit


class WaveformAugmentation(StochasticAudioPreprocessingUnit, ABC):
    ...


class RandomGain(WaveformAugmentation):
    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0, p: float = 0.5):
        super().__init__(p)
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def _apply(self, waveform: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain = 10.0 ** (gain_db / 20.0)

        return (waveform * gain).clamp_(-1.0, 1.0)


class AdditiveNoise(WaveformAugmentation):
    def __init__(self, snr_db_min: float = 10.0, snr_db_max: float = 30.0, p: float = 0.5):
        super().__init__(p)
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max

    def _apply(self, waveform: torch.Tensor) -> torch.Tensor:
        signal_power = waveform.pow(2).mean().item()
        if signal_power <= 1e-12:
            return waveform

        snr_db = random.uniform(self.snr_db_min, self.snr_db_max)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = max(signal_power / snr_linear, 1e-12)
        noise = torch.randn_like(waveform) * (noise_power**0.5)

        return (waveform + noise).clamp_(-1.0, 1.0)


class RandomTimeShift(WaveformAugmentation):
    def __init__(
        self, max_shift_seconds: float = 0.1, sample_rate: int = 16000, p: float = 0.5
    ):
        super().__init__(p)
        self.max_shift = int(max_shift_seconds * sample_rate)

    def _apply(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return waveform

        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return waveform

        ch, length = waveform.shape[-2], waveform.shape[-1]
        if shift > 0:
            pad = torch.zeros(ch, shift, dtype=waveform.dtype, device=waveform.device)
            waveform = torch.cat([pad, waveform], dim=-1)[..., :length]
        else:
            pad = torch.zeros(ch, -shift, dtype=waveform.dtype, device=waveform.device)
            waveform = torch.cat([waveform, pad], dim=-1)[..., -length:]

        return waveform
