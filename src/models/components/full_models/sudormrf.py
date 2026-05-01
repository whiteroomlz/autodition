from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from src.models.components.base import (
    HeadStage,
    ModelContext,
    Ref,
    TensorSlot,
    ensure_single_input,
)
from src.models.components.separation import project_sources_to_mixture


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.act = nn.PReLU(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(inputs)))


class _ConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(1, out_channels, eps=1e-8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(inputs))


class _NormAct(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=1e-8)
        self.act = nn.PReLU(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(inputs))


class _DilatedConvNorm(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels, eps=1e-8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(inputs))


class _UBlock(nn.Module):
    def __init__(self, out_channels: int, bottleneck_channels: int, upsampling_depth: int):
        super().__init__()
        self.depth = upsampling_depth
        self.project = _ConvNormAct(out_channels, bottleneck_channels, 1)
        self.depthwise_blocks = nn.ModuleList(
            [
                _DilatedConvNorm(bottleneck_channels, kernel_size=5, stride=1),
                *[
                    _DilatedConvNorm(
                        bottleneck_channels,
                        kernel_size=(2 * 2) + 1,
                        stride=2,
                    )
                    for _ in range(1, upsampling_depth)
                ],
            ]
        )
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest") if upsampling_depth > 1 else None
        )
        self.expand = _ConvNorm(bottleneck_channels, out_channels, 1)
        self.final_norm = _NormAct(bottleneck_channels)
        self.output_act = _NormAct(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.project(inputs)
        pyramid = [self.depthwise_blocks[0](output)]
        for block in self.depthwise_blocks[1:]:
            pyramid.append(block(pyramid[-1]))

        while len(pyramid) > 1:
            upsampled = self.upsample(pyramid.pop(-1))
            if upsampled.shape[-1] != pyramid[-1].shape[-1]:
                upsampled = upsampled[..., : pyramid[-1].shape[-1]]
            pyramid[-1] = pyramid[-1] + upsampled

        expanded = self.expand(self.final_norm(pyramid[-1]))
        return self.output_act(expanded + inputs)


class _SuDORMRFCore(nn.Module):
    def __init__(
        self,
        out_channels: int,
        bottleneck_channels: int,
        num_blocks: int,
        upsampling_depth: int,
        enc_kernel_size: int,
        enc_num_basis: int,
        num_sources: int,
        decoder_init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.enc_kernel_size = enc_kernel_size
        self.upsampling_depth = upsampling_depth
        self.num_sources = num_sources
        self.lcm = abs(enc_kernel_size // 2 * 2**upsampling_depth) // math.gcd(
            enc_kernel_size // 2,
            2**upsampling_depth,
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=enc_num_basis,
                kernel_size=enc_kernel_size,
                stride=enc_kernel_size // 2,
                padding=enc_kernel_size // 2,
            ),
            nn.ReLU(),
        )
        self.norm = nn.GroupNorm(1, enc_num_basis, eps=1e-8)
        self.project = nn.Conv1d(enc_num_basis, out_channels, kernel_size=1)
        self.separator = nn.Sequential(
            *[
                _UBlock(
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    upsampling_depth=upsampling_depth,
                )
                for _ in range(num_blocks)
            ]
        )
        self.reshape_before_masks = (
            nn.Conv1d(out_channels, enc_num_basis, kernel_size=1)
            if out_channels != enc_num_basis
            else None
        )
        self.mask_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_sources,
            kernel_size=(enc_num_basis + 1, 1),
            padding=(enc_num_basis - enc_num_basis // 2, 0),
        )
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=num_sources,
        )
        if decoder_init_gain != 1.0:
            nn.init.xavier_uniform_(self.decoder.weight, gain=decoder_init_gain)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        original_length = waveform.shape[-1]
        waveform = self._pad_to_appropriate_length(waveform)
        encoded = self.encoder(waveform)
        skip = encoded.clone()

        separated = self.project(self.norm(encoded))
        separated = self.separator(separated)
        if self.reshape_before_masks is not None:
            separated = self.reshape_before_masks(separated)

        masks = self.mask_conv(separated.unsqueeze(1))
        masks = torch.softmax(masks, dim=1)
        masked = masks * skip.unsqueeze(1)
        estimated = self.decoder(masked.view(masked.shape[0], -1, masked.shape[-1]))
        return estimated[..., :original_length]

    def _pad_to_appropriate_length(self, waveform: torch.Tensor) -> torch.Tensor:
        padding = waveform.shape[-1] % self.lcm
        if padding == 0:
            return waveform

        padded_length = waveform.shape[-1] + self.lcm - padding
        padded_waveform = waveform.new_zeros(waveform.shape[0], waveform.shape[1], padded_length)
        padded_waveform[..., : waveform.shape[-1]] = waveform
        return padded_waveform


class SuDORMRFSeparator(HeadStage):
    """Field-native waveform separator based on SuDoRM-RF."""

    def __init__(
        self,
        inputs: Sequence[Ref],
        output_name: str,
        num_sources: int = 4,
        out_channels: int = 128,
        bottleneck_channels: int = 256,
        num_blocks: int = 8,
        upsampling_depth: int = 4,
        enc_kernel_size: int = 21,
        enc_num_basis: int = 256,
        enforce_mixture_consistency: bool = False,
        mixture_residual_connection: bool = False,
        decoder_init_gain: float = 1.0,
    ) -> None:
        super().__init__(inputs=inputs, outputs=(output_name,))
        self.enforce_mixture_consistency = enforce_mixture_consistency
        self.mixture_residual_connection = mixture_residual_connection
        self.core = _SuDORMRFCore(
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            enc_kernel_size=enc_kernel_size,
            enc_num_basis=enc_num_basis,
            num_sources=num_sources,
            decoder_init_gain=decoder_init_gain,
        )

    def forward(self, context: ModelContext) -> ModelContext:
        slot = ensure_single_input(
            tuple(context.resolve_slot(ref) for ref in self.inputs),
            self.__class__.__name__,
        )
        if slot.value.ndim != 2:
            raise ValueError("SuDORMRFSeparator expects waveform tensors shaped [batch, time]")

        waveform = slot.value.unsqueeze(1)
        estimated_sources = self.core(waveform)
        estimated_sources = estimated_sources[..., : slot.value.shape[-1]]
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
