"""Source-separation-specific criteria and loss terms."""

from __future__ import annotations

import inspect
from typing import Dict, Sequence

import torch

from src.data.components.batch import Batch

from ..base import ModelContext, ModelResult, Ref
from ..separation import (
    build_best_permutation,
    compute_si_sdr,
    reduce_loss_over_nonbatch_dims,
    sum_sources,
)
from .base import Criterion, LossTerm, LossWeight, coerce_optional_ref, coerce_ref
from .generic import L1Criterion, MeanSquaredErrorCriterion


class NegativeSISDRCriterion(Criterion):
    """Negative SI-SDR reduced per sample for waveform separation."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        return -compute_si_sdr(prediction, target, mask=mask, eps=self.eps)


class MultiResolutionSTFTCriterion(Criterion):
    """Multi-resolution STFT reconstruction loss reduced per sample."""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (256, 512, 1024),
        hop_sizes: Sequence[int] = (64, 128, 256),
        win_lengths: Sequence[int] = (256, 512, 1024),
        spectral_convergence_weight: float = 0.5,
        log_magnitude_weight: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length")

        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.hop_sizes = tuple(int(value) for value in hop_sizes)
        self.win_lengths = tuple(int(value) for value in win_lengths)
        self.spectral_convergence_weight = spectral_convergence_weight
        self.log_magnitude_weight = log_magnitude_weight
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("MultiResolutionSTFTCriterion expects matching prediction/target")
        if prediction.ndim != 2:
            raise ValueError("MultiResolutionSTFTCriterion expects tensors shaped [batch, time]")

        if mask is not None:
            mask = mask.to(device=prediction.device, dtype=prediction.dtype)
            prediction = prediction * mask
            target = target * mask

        total_loss = prediction.new_zeros(prediction.shape[0])
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes,
            self.hop_sizes,
            self.win_lengths,
        ):
            window = torch.hann_window(
                win_length, device=prediction.device, dtype=prediction.dtype
            )
            prediction_spec = torch.stft(
                prediction,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
            target_spec = torch.stft(
                target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            prediction_mag = prediction_spec.abs().clamp_min(self.eps)
            target_mag = target_spec.abs().clamp_min(self.eps)

            spectral_convergence = (target_mag - prediction_mag).square().sum(
                dim=(-2, -1)
            ).sqrt() / target_mag.square().sum(dim=(-2, -1)).sqrt().clamp_min(self.eps)
            log_magnitude = (target_mag.log() - prediction_mag.log()).abs().mean(dim=(-2, -1))
            total_loss = total_loss + (
                self.spectral_convergence_weight * spectral_convergence
                + self.log_magnitude_weight * log_magnitude
            )

        return total_loss / len(self.fft_sizes)


class PermutationInvariantLossTerm(LossTerm):
    """Permutation-invariant loss for fixed-slot source separation outputs."""

    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        criterion: Criterion,
        activity_ref: Ref | Dict[str, str] | None = None,
        mask_ref: Ref | Dict[str, str] | None = None,
        inactive_criterion: Criterion | None = None,
        inactive_weight: float = 1.0,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__(name=name, loss_weight=loss_weight)
        self.prediction_ref = coerce_ref(prediction_ref)
        self.target_ref = coerce_ref(target_ref)
        self.activity_ref = coerce_optional_ref(activity_ref)
        self.mask_ref = coerce_optional_ref(mask_ref)
        self.criterion = criterion
        self.inactive_criterion = inactive_criterion or MeanSquaredErrorCriterion()
        self.inactive_weight = inactive_weight
        self._criterion_accepts_mask = _criterion_accepts_mask(criterion)

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: int | None = None,
    ) -> torch.Tensor:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        target = context.resolve_tensor(self.target_ref)
        time_mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        activity = (
            context.resolve_tensor(self.activity_ref).bool()
            if self.activity_ref is not None
            else torch.ones(
                target.shape[0],
                target.shape[1],
                dtype=torch.bool,
                device=target.device,
            )
        )

        if prediction.shape != target.shape:
            raise ValueError(
                "PermutationInvariantLossTerm expects matching prediction/target shapes"
            )
        if prediction.ndim != 3:
            raise ValueError(
                "PermutationInvariantLossTerm expects tensors shaped [batch, source, time]"
            )

        pairwise_active = []
        pairwise_inactive = []
        for prediction_index in range(prediction.shape[1]):
            active_row = []
            inactive_row = []
            for target_index in range(target.shape[1]):
                active_row.append(
                    reduce_loss_over_nonbatch_dims(
                        self._call_active_criterion(
                            prediction[:, prediction_index, :],
                            target[:, target_index, :],
                            time_mask,
                        ),
                        mask=time_mask,
                    )
                )
                inactive_row.append(
                    reduce_loss_over_nonbatch_dims(
                        self.inactive_criterion(
                            prediction[:, prediction_index, :],
                            target.new_zeros(target[:, target_index, :].shape),
                        ),
                        mask=time_mask,
                    )
                )
            pairwise_active.append(torch.stack(active_row, dim=1))
            pairwise_inactive.append(torch.stack(inactive_row, dim=1))

        active_cost = torch.stack(pairwise_active, dim=1)
        inactive_cost = torch.stack(pairwise_inactive, dim=1)
        pairwise_cost = torch.where(
            activity[:, None, :],
            active_cost,
            inactive_cost * self.inactive_weight,
        )

        _, best_costs = build_best_permutation(pairwise_cost)
        return best_costs.mean() * self.loss_weight(step)

    def _call_active_criterion(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.BoolTensor | None,
    ) -> torch.Tensor:
        if self._criterion_accepts_mask:
            return self.criterion(prediction, target, mask=mask)
        return self.criterion(prediction, target)


class SummedSourcesConsistencyLossTerm(LossTerm):
    """Consistency loss between the source sum and a mixture reference."""

    def __init__(
        self,
        name: str,
        sources_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        criterion: Criterion,
        mask_ref: Ref | Dict[str, str] | None = None,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__(name=name, loss_weight=loss_weight)
        self.sources_ref = coerce_ref(sources_ref)
        self.target_ref = coerce_ref(target_ref)
        self.mask_ref = coerce_optional_ref(mask_ref)
        self.criterion = criterion

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: int | None = None,
    ) -> torch.Tensor:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        sources = context.resolve_tensor(self.sources_ref)
        target = context.resolve_tensor(self.target_ref)
        mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        summed_sources = sum_sources(sources)
        base_loss = self.criterion(summed_sources, target)
        reduced = reduce_loss_over_nonbatch_dims(base_loss, mask=mask).mean()
        return reduced * self.loss_weight(step)


def _criterion_accepts_mask(criterion: Criterion) -> bool:
    try:
        signature = inspect.signature(criterion.forward)
    except (TypeError, ValueError):
        return False

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == "mask"
        for name, parameter in signature.parameters.items()
    )
