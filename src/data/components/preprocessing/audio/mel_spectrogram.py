from typing import Optional

import torch
import torchaudio

from .base import AudioPreprocessingUnit


class MelSpectrogram(AudioPreprocessingUnit):
    """Compute log-mel spectrogram from a waveform tensor.

        C - channels (1 for mono)
        T - time frames
        F - frequency bins (n_mels)
    Args:
        waveform: C x samples -> T x F
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        )

    def _apply(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        :param waveform: C x samples
        :return: T x F (squeezed from C x F x T, transposed)
        """
        mel = self.mel(waveform)
        log_mel = torch.log(mel + 1e-9)

        return log_mel.squeeze(0).transpose(0, 1)
