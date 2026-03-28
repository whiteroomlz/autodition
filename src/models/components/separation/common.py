"""Shared helpers for source-separation objectives, metrics, and stages."""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional

import torch


def reduce_loss_over_nonbatch_dims(
    loss: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """Reduce a loss tensor to one scalar per batch item."""

    if loss.ndim == 0:
        return loss.unsqueeze(0)

    if loss.ndim == 1:
        return loss

    if mask is None:
        return loss.reshape(loss.shape[0], -1).mean(dim=1)

    mask = mask.to(device=loss.device, dtype=loss.dtype)
    expanded_mask = _broadcast_mask(mask, loss)
    weighted_loss = loss * expanded_mask
    denominator = expanded_mask.reshape(expanded_mask.shape[0], -1).sum(dim=1)
    numerator = weighted_loss.reshape(weighted_loss.shape[0], -1).sum(dim=1)
    safe_denominator = denominator.clamp_min(1.0)
    reduced = numerator / safe_denominator
    return torch.where(denominator > 0, reduced, numerator * 0.0)


def compute_si_sdr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute SI-SDR per batch item for tensors shaped ``[batch, time]``."""

    if prediction.shape != target.shape:
        raise ValueError("SI-SDR expects prediction and target with matching shapes")
    if prediction.ndim != 2:
        raise ValueError("SI-SDR expects tensors shaped [batch, time]")

    if mask is not None:
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)
        prediction = prediction * mask
        target = target * mask
        valid_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        prediction = prediction - prediction.sum(dim=1, keepdim=True) / valid_lengths
        target = target - target.sum(dim=1, keepdim=True) / valid_lengths
    else:
        prediction = prediction - prediction.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)

    target_energy = target.square().sum(dim=1, keepdim=True).clamp_min(eps)
    projection = (prediction * target).sum(dim=1, keepdim=True) * target / target_energy
    noise = prediction - projection
    ratio = projection.square().sum(dim=1) / noise.square().sum(dim=1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def build_best_permutation(
    pairwise_cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the minimum-cost assignment over the source axis."""

    if pairwise_cost.ndim != 3:
        raise ValueError("pairwise_cost must have shape [batch, prediction, target]")

    _, num_predictions, num_targets = pairwise_cost.shape
    if num_predictions != num_targets:
        raise ValueError("pairwise_cost must be square on the source dimensions")

    permutations = _source_permutations(num_predictions, pairwise_cost.device)
    target_indices = torch.arange(num_targets, device=pairwise_cost.device)
    permutation_costs = torch.stack(
        [pairwise_cost[:, permutation, target_indices].sum(dim=1) for permutation in permutations],
        dim=1,
    )
    best_indices = permutation_costs.argmin(dim=1)
    best_permutations = permutations[best_indices]
    best_costs = permutation_costs.gather(1, best_indices[:, None]).squeeze(1)
    return best_permutations, best_costs


def align_sources(
    sources: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Align sources using a per-example permutation tensor."""

    if sources.ndim < 3:
        raise ValueError("sources must have at least shape [batch, source, ...]")
    if permutation.ndim != 2:
        raise ValueError("permutation must have shape [batch, source]")
    if sources.shape[0] != permutation.shape[0] or sources.shape[1] != permutation.shape[1]:
        raise ValueError("permutation shape is incompatible with sources")

    expand_shape = [sources.shape[0], sources.shape[1], *([1] * (sources.ndim - 2))]
    gather_index = permutation.view(*expand_shape).expand(-1, -1, *sources.shape[2:])
    return sources.gather(dim=1, index=gather_index)


def sum_sources(sources: torch.Tensor) -> torch.Tensor:
    """Collapse the explicit source axis back to one waveform per sample."""

    if sources.ndim < 3:
        raise ValueError("sources must have at least shape [batch, source, time]")
    return sources.sum(dim=1)


def infer_source_activity(
    sources: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    eps: float = 1e-8,
) -> torch.BoolTensor:
    """Infer which sources are active from target energy."""

    if sources.ndim < 3:
        raise ValueError("sources must have at least shape [batch, source, time]")

    if mask is not None:
        mask = mask.to(device=sources.device, dtype=sources.dtype)
        while mask.ndim < sources.ndim:
            mask = mask.unsqueeze(1)
        sources = sources * mask

    energy = sources.reshape(sources.shape[0], sources.shape[1], -1).square().sum(dim=-1)
    return energy > eps


def flatten_active_sources(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.BoolTensor,
    mask: torch.BoolTensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor | None]:
    """Flatten active sources across the batch for metric computation."""

    active_prediction = prediction[activity]
    active_target = target[activity]
    if mask is None:
        return active_prediction, active_target, None

    active_mask = mask[:, None, :].expand_as(target)[activity]
    return active_prediction, active_target, active_mask


def _broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.shape == target.shape:
        return mask

    candidate = mask
    while candidate.ndim < target.ndim:
        candidate = candidate.unsqueeze(1)

    try:
        return torch.broadcast_to(candidate, target.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Cannot broadcast mask with shape {tuple(mask.shape)} "
            f"to target shape {tuple(target.shape)}"
        ) from error


@lru_cache(maxsize=None)
def _cached_source_permutations(num_sources: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(num_sources)))


def _source_permutations(num_sources: int, device: torch.device) -> torch.Tensor:
    permutations = _cached_source_permutations(num_sources)
    return torch.tensor(permutations, dtype=torch.long, device=device)
