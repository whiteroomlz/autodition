"""Objective orchestration and shared base contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.components.batch import Batch

from ..base import ModelResult, Ref


class LossWeight(ABC):
    """Resolve a scalar multiplier for a loss term, optionally step-dependent."""

    @abstractmethod
    def __call__(self, step: Optional[int] = None) -> float:
        raise NotImplementedError


class ConstantLossWeight(LossWeight):
    """Fixed scalar multiplier used by default for most loss terms."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def __call__(self, step: Optional[int] = None) -> float:
        return self.value


class Criterion(torch.nn.Module, ABC):
    """Narrow criterion contract that returns unreduced per-element loss values."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LossTerm(torch.nn.Module, ABC):
    """Single named loss node that reads refs and returns one reduced scalar."""

    def __init__(
        self,
        name: str,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.loss_weight = loss_weight or ConstantLossWeight()

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _apply_postprocessing(
        self,
        base_loss: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        loss = base_loss
        if loss.ndim == 0:
            if mask is not None or weight is not None:
                raise ValueError(
                    f"Loss term '{self.name}' received scalar loss together with mask/weight"
                )
            return loss * self.loss_weight(step)

        scale = torch.ones_like(loss, dtype=loss.dtype)
        if mask is not None:
            scale = scale * self._broadcast(mask.to(loss.device, dtype=loss.dtype), loss)
        if weight is not None:
            scale = scale * self._broadcast(weight.to(loss.device, dtype=loss.dtype), loss)

        weighted_loss = loss * scale
        denominator = scale.sum()
        if denominator.item() <= 0:
            reduced = weighted_loss.sum() * 0.0
        else:
            reduced = weighted_loss.sum() / denominator
        return reduced * self.loss_weight(step)

    @staticmethod
    def _broadcast(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape == target.shape:
            return source

        candidates = [source]
        while candidates:
            candidate = candidates.pop(0)
            if candidate.ndim == target.ndim:
                try:
                    return torch.broadcast_to(candidate, target.shape)
                except RuntimeError:
                    continue

            if candidate.ndim < target.ndim:
                for dim in range(candidate.ndim + 1):
                    candidates.append(candidate.unsqueeze(dim))

        raise ValueError(
            f"Cannot broadcast tensor with shape {tuple(source.shape)} "
            f"to target shape {tuple(target.shape)}"
        )


class ObjectiveComposer(torch.nn.Module):
    """Aggregate configured loss terms into total loss plus per-term breakdown."""

    def __init__(self, terms: Sequence[LossTerm]) -> None:
        super().__init__()
        self.terms = torch.nn.ModuleList(terms)

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        total_loss: Optional[torch.Tensor] = None

        for term in self.terms:
            term_loss = term(batch=batch, result=result, step=step)
            losses[term.name] = term_loss
            total_loss = term_loss if total_loss is None else total_loss + term_loss

        if total_loss is None:
            raise ValueError("ObjectiveComposer requires at least one loss term")

        return total_loss, losses


def coerce_ref(ref: Ref | Dict[str, str]) -> Ref:
    if isinstance(ref, DictConfig):
        ref = OmegaConf.to_object(ref)
    if isinstance(ref, Ref):
        return ref
    if isinstance(ref, dict):
        ref_dict = {key: value for key, value in ref.items() if not key.startswith("_")}
        return Ref(**ref_dict)
    raise TypeError(f"Unsupported ref type: {type(ref).__name__}")


def coerce_optional_ref(ref: Ref | Dict[str, str] | None) -> Ref | None:
    if ref is None:
        return None
    return coerce_ref(ref)
