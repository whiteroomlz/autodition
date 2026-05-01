from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from src.models.components.base import (
    HeadStage,
    ModelContext,
    Ref,
    TensorSlot,
    ensure_single_input,
)
from src.models.components.separation import project_sources_to_mixture


def _activation(name: str, channels: int | None = None) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name == "prelu":
        return nn.PReLU(channels or 1)
    if normalized_name == "relu":
        return nn.ReLU()
    if normalized_name == "elu":
        return nn.ELU()
    if normalized_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


class _LayerNormalization4D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = Parameter(torch.ones(1, channels, 1, 1, dtype=torch.float32))
        self.beta = Parameter(torch.zeros(1, channels, 1, 1, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(1,), keepdim=True)
        std = torch.sqrt(inputs.var(dim=(1,), unbiased=False, keepdim=True) + self.eps)
        return ((inputs - mean) / std) * self.gamma + self.beta


class _LayerNormalization4DCF(nn.Module):
    def __init__(self, channels: int, num_freqs: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = Parameter(torch.ones(1, channels, 1, num_freqs, dtype=torch.float32))
        self.beta = Parameter(torch.zeros(1, channels, 1, num_freqs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(1, 3), keepdim=True)
        std = torch.sqrt(inputs.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps)
        return ((inputs - mean) / std) * self.gamma + self.beta


class _AttentionProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_freqs: int, activation: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            _activation(activation, out_channels),
            _LayerNormalization4DCF(out_channels, num_freqs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _GridNetBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        emb_kernel_size: int,
        emb_hop_size: int,
        num_freqs: int,
        hidden_channels: int,
        num_heads: int = 4,
        approx_qk_dim: int = 256,
        activation: str = "prelu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        in_channels = emb_dim * emb_kernel_size

        self.intra_norm = _LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2,
            emb_dim,
            emb_kernel_size,
            stride=emb_hop_size,
        )

        self.inter_norm = _LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2,
            emb_dim,
            emb_kernel_size,
            stride=emb_hop_size,
        )

        head_qk_dim = math.ceil(approx_qk_dim / num_freqs)
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        self.query_projections = nn.ModuleList(
            [
                _AttentionProjection(emb_dim, head_qk_dim, num_freqs, activation)
                for _ in range(num_heads)
            ]
        )
        self.key_projections = nn.ModuleList(
            [
                _AttentionProjection(emb_dim, head_qk_dim, num_freqs, activation)
                for _ in range(num_heads)
            ]
        )
        self.value_projections = nn.ModuleList(
            [
                _AttentionProjection(emb_dim, emb_dim // num_heads, num_freqs, activation)
                for _ in range(num_heads)
            ]
        )
        self.concat_projection = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1),
            _activation(activation, emb_dim),
            _LayerNormalization4DCF(emb_dim, num_freqs, eps=eps),
        )
        self.emb_dim = emb_dim
        self.emb_kernel_size = emb_kernel_size
        self.emb_hop_size = emb_hop_size
        self.num_heads = num_heads

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, original_time, original_freq = inputs.shape
        padded_time = (
            math.ceil((original_time - self.emb_kernel_size) / self.emb_hop_size)
            * self.emb_hop_size
            + self.emb_kernel_size
        )
        padded_freq = (
            math.ceil((original_freq - self.emb_kernel_size) / self.emb_hop_size)
            * self.emb_hop_size
            + self.emb_kernel_size
        )
        x = F.pad(inputs, (0, padded_freq - original_freq, 0, padded_time - original_time))

        intra_inputs = (
            self.intra_norm(x)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * padded_time, channels, padded_freq)
        )
        intra_inputs = F.unfold(
            intra_inputs[..., None],
            (self.emb_kernel_size, 1),
            stride=(self.emb_hop_size, 1),
        ).transpose(1, 2)
        intra_outputs, _ = self.intra_rnn(intra_inputs)
        intra_outputs = self.intra_linear(intra_outputs.transpose(1, 2))
        intra_outputs = (
            intra_outputs.view(batch_size, padded_time, channels, padded_freq)
            .transpose(1, 2)
            .contiguous()
        )
        intra_outputs = intra_outputs + x

        inter_inputs = (
            self.inter_norm(intra_outputs)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * padded_freq, channels, padded_time)
        )
        inter_inputs = F.unfold(
            inter_inputs[..., None],
            (self.emb_kernel_size, 1),
            stride=(self.emb_hop_size, 1),
        ).transpose(1, 2)
        inter_outputs, _ = self.inter_rnn(inter_inputs)
        inter_outputs = self.inter_linear(inter_outputs.transpose(1, 2))
        inter_outputs = (
            inter_outputs.view(batch_size, padded_freq, channels, padded_time)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        inter_outputs = inter_outputs + intra_outputs
        inter_outputs = inter_outputs[..., :original_time, :original_freq]

        queries = torch.cat(
            [projection(inter_outputs) for projection in self.query_projections], dim=0
        )
        keys = torch.cat([projection(inter_outputs) for projection in self.key_projections], dim=0)
        values = torch.cat(
            [projection(inter_outputs) for projection in self.value_projections], dim=0
        )

        queries = queries.transpose(1, 2).flatten(start_dim=2)
        keys = keys.transpose(1, 2).flatten(start_dim=2)
        values = values.transpose(1, 2)
        original_value_shape = values.shape
        values = values.flatten(start_dim=2)

        attention = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])
        attention = torch.softmax(attention, dim=-1)
        values = torch.matmul(attention, values)
        values = values.reshape(original_value_shape).transpose(1, 2)

        values = values.view(self.num_heads, batch_size, values.shape[1], original_time, -1)
        values = (
            values.transpose(0, 1).contiguous().view(batch_size, self.emb_dim, original_time, -1)
        )
        outputs = self.concat_projection(values)
        return outputs + inter_outputs


class _TFGridNetCore(nn.Module):
    def __init__(
        self,
        num_sources: int,
        n_fft: int,
        hop_length: int,
        num_layers: int,
        lstm_hidden_units: int,
        emb_dim: int,
        emb_kernel_size: int,
        emb_hop_size: int,
        num_heads: int,
        approx_qk_dim: int,
        activation: str,
        eps: float,
        output_projection_init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if n_fft % 2 != 0:
            raise ValueError("n_fft must be even")

        self.num_sources = num_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = n_fft
        self.eps = eps
        self.num_freqs = (n_fft // 2) + 1

        self.input_projection = nn.Sequential(
            nn.Conv2d(2, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList(
            [
                _GridNetBlock(
                    emb_dim=emb_dim,
                    emb_kernel_size=emb_kernel_size,
                    emb_hop_size=emb_hop_size,
                    num_freqs=self.num_freqs,
                    hidden_channels=lstm_hidden_units,
                    num_heads=num_heads,
                    approx_qk_dim=approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.ConvTranspose2d(
            emb_dim,
            num_sources * 2,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        if output_projection_init_gain != 1.0:
            nn.init.xavier_uniform_(
                self.output_projection.weight, gain=output_projection_init_gain
            )
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        original_length = waveform.shape[-1]
        mixture_std = waveform.std(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_waveform = waveform / mixture_std

        window = torch.hann_window(
            self.window_length,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        mixture_spec = torch.stft(
            normalized_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=window,
            return_complex=True,
        )
        features = torch.stack((mixture_spec.real, mixture_spec.imag), dim=1).permute(0, 1, 3, 2)
        features = self.input_projection(features)
        for block in self.blocks:
            features = block(features)

        separated = self.output_projection(features)
        batch_size, _, num_frames, num_freqs = separated.shape
        separated = separated.view(batch_size, self.num_sources, 2, num_frames, num_freqs)
        separated = torch.complex(
            separated[:, :, 0].permute(0, 1, 3, 2).contiguous(),
            separated[:, :, 1].permute(0, 1, 3, 2).contiguous(),
        )

        estimated = torch.istft(
            separated.view(batch_size * self.num_sources, num_freqs, num_frames),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=window,
            length=original_length,
        )
        estimated = estimated.view(batch_size, self.num_sources, original_length)
        return estimated * mixture_std.unsqueeze(1)


class TFGridNetSeparator(HeadStage):
    """Field-native compact TF-GridNet separator."""

    def __init__(
        self,
        inputs: Sequence[Ref],
        output_name: str,
        num_sources: int = 4,
        n_fft: int = 512,
        hop_length: int = 128,
        num_layers: int = 4,
        lstm_hidden_units: int = 96,
        emb_dim: int = 32,
        emb_kernel_size: int = 4,
        emb_hop_size: int = 1,
        num_heads: int = 4,
        approx_qk_dim: int = 256,
        activation: str = "prelu",
        eps: float = 1e-5,
        enforce_mixture_consistency: bool = False,
        mixture_residual_connection: bool = False,
        output_projection_init_gain: float = 1.0,
    ) -> None:
        super().__init__(inputs=inputs, outputs=(output_name,))
        self.enforce_mixture_consistency = enforce_mixture_consistency
        self.mixture_residual_connection = mixture_residual_connection
        self.core = _TFGridNetCore(
            num_sources=num_sources,
            n_fft=n_fft,
            hop_length=hop_length,
            num_layers=num_layers,
            lstm_hidden_units=lstm_hidden_units,
            emb_dim=emb_dim,
            emb_kernel_size=emb_kernel_size,
            emb_hop_size=emb_hop_size,
            num_heads=num_heads,
            approx_qk_dim=approx_qk_dim,
            activation=activation,
            eps=eps,
            output_projection_init_gain=output_projection_init_gain,
        )

    def forward(self, context: ModelContext) -> ModelContext:
        slot = ensure_single_input(
            tuple(context.resolve_slot(ref) for ref in self.inputs),
            self.__class__.__name__,
        )
        if slot.value.ndim != 2:
            raise ValueError("TFGridNetSeparator expects waveform tensors shaped [batch, time]")

        estimated_sources = self.core(slot.value)
        if self.mixture_residual_connection:
            estimated_sources = estimated_sources.clone()
            estimated_sources[:, 0, :] = estimated_sources[:, 0, :] + slot.value
        if slot.mask is not None:
            estimated_sources = estimated_sources * slot.mask[:, None, :].to(
                dtype=estimated_sources.dtype
            )
        if self.enforce_mixture_consistency:
            estimated_sources = project_sources_to_mixture(
                estimated_sources,
                slot.value,
                mask=slot.mask,
            )

        context.write(
            self.store,
            self.outputs[0],
            TensorSlot(value=estimated_sources, mask=slot.mask),
        )
        return context
