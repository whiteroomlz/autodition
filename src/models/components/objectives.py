from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.data.components.batch import Batch

from .base import ModelContext, ModelResult, Ref


class LossWeight(ABC):
    @abstractmethod
    def __call__(self, step: Optional[int] = None) -> float:
        raise NotImplementedError


class ConstantLossWeight(LossWeight):
    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def __call__(self, step: Optional[int] = None) -> float:
        return self.value


class Criterion(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class CrossEntropyCriterion(Criterion):
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            prediction,
            target.long(),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )


class MeanSquaredErrorCriterion(Criterion):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target, reduction="none")


class L1Criterion(Criterion):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(prediction, target, reduction="none")


class KLDivCriterion(Criterion):
    def __init__(self, reduction: str = "none", log_target: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prediction = F.log_softmax(prediction, dim=-1)
        normalized_target = target if self.log_target else F.softmax(target, dim=-1)
        loss = F.kl_div(
            log_prediction,
            normalized_target,
            reduction=self.reduction,
            log_target=self.log_target,
        )
        if self.reduction == "none":
            return loss
        return torch.as_tensor(loss, device=prediction.device)


class SquaredL2Criterion(Criterion):
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source.square()


class LossTerm(torch.nn.Module, ABC):
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


class SupervisedLossTerm(LossTerm):
    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        criterion: Criterion,
        mask_ref: Ref | Dict[str, str] | None = None,
        weight_ref: Ref | Dict[str, str] | None = None,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__(name=name, loss_weight=loss_weight)
        self.prediction_ref = _coerce_ref(prediction_ref)
        self.target_ref = _coerce_ref(target_ref)
        self.criterion = criterion
        self.mask_ref = _coerce_optional_ref(mask_ref)
        self.weight_ref = _coerce_optional_ref(weight_ref)

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        target = context.resolve_tensor(self.target_ref)
        mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        weight = context.resolve_tensor(self.weight_ref) if self.weight_ref is not None else None
        base_loss = self.criterion(prediction, target)
        return self._apply_postprocessing(base_loss, mask=mask, weight=weight, step=step)


class ConsistencyLossTerm(LossTerm):
    def __init__(
        self,
        name: str,
        left_ref: Ref | Dict[str, str],
        right_ref: Ref | Dict[str, str],
        criterion: Criterion,
        mask_ref: Ref | Dict[str, str] | None = None,
        weight_ref: Ref | Dict[str, str] | None = None,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__(name=name, loss_weight=loss_weight)
        self.left_ref = _coerce_ref(left_ref)
        self.right_ref = _coerce_ref(right_ref)
        self.criterion = criterion
        self.mask_ref = _coerce_optional_ref(mask_ref)
        self.weight_ref = _coerce_optional_ref(weight_ref)

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        left = context.resolve_tensor(self.left_ref)
        right = context.resolve_tensor(self.right_ref)
        mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        weight = context.resolve_tensor(self.weight_ref) if self.weight_ref is not None else None
        base_loss = self.criterion(left, right)
        return self._apply_postprocessing(base_loss, mask=mask, weight=weight, step=step)


class RegularizationLossTerm(LossTerm):
    def __init__(
        self,
        name: str,
        source_ref: Ref | Dict[str, str],
        criterion: Criterion,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__(name=name, loss_weight=loss_weight)
        self.source_ref = _coerce_ref(source_ref)
        self.criterion = criterion

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)
        source = context.resolve_tensor(self.source_ref)
        base_loss = self.criterion(source)
        return self._apply_postprocessing(base_loss, step=step)


class ObjectiveComposer(torch.nn.Module):
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


def _coerce_ref(ref: Ref | Dict[str, str]) -> Ref:
    if isinstance(ref, DictConfig):
        ref = OmegaConf.to_object(ref)
    if isinstance(ref, Ref):
        return ref
    if isinstance(ref, dict):
        ref_dict = {key: value for key, value in ref.items() if not key.startswith("_")}
        return Ref(**ref_dict)
    raise TypeError(f"Unsupported ref type: {type(ref).__name__}")


def _coerce_optional_ref(ref: Ref | Dict[str, str] | None) -> Ref | None:
    if ref is None:
        return None
    return _coerce_ref(ref)
