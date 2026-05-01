"""Generic task losses and loss terms."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.data.components.batch import Batch

from ..base import ModelContext, ModelResult, Ref
from .base import Criterion, LossTerm, LossWeight, coerce_optional_ref, coerce_ref


class CrossEntropyCriterion(Criterion):
    """Cross-entropy over class logits and integer targets."""

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
    """Elementwise mean squared error."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target, reduction="none")


class L1Criterion(Criterion):
    """Elementwise absolute error."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(prediction, target, reduction="none")


class KLDivCriterion(Criterion):
    """KL divergence for distillation-style logits or distributions."""

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
    """Squared L2 penalty used for simple regularization terms."""

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source.square()


class SupervisedLossTerm(LossTerm):
    """Loss between a prediction ref and a supervision ref."""

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
        self.prediction_ref = coerce_ref(prediction_ref)
        self.target_ref = coerce_ref(target_ref)
        self.criterion = criterion
        self.mask_ref = coerce_optional_ref(mask_ref)
        self.weight_ref = coerce_optional_ref(weight_ref)

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
        mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        weight = context.resolve_tensor(self.weight_ref) if self.weight_ref is not None else None
        base_loss = self.criterion(prediction, target)
        return self._apply_postprocessing(base_loss, mask=mask, weight=weight, step=step)


class ConsistencyLossTerm(LossTerm):
    """Loss enforcing agreement between two model-side tensors."""

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
        self.left_ref = coerce_ref(left_ref)
        self.right_ref = coerce_ref(right_ref)
        self.criterion = criterion
        self.mask_ref = coerce_optional_ref(mask_ref)
        self.weight_ref = coerce_optional_ref(weight_ref)

    def forward(
        self,
        batch: Batch,
        result: ModelResult,
        step: int | None = None,
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
    """Loss computed from one source tensor without an explicit supervision target."""

    def __init__(
        self,
        name: str,
        source_ref: Ref | Dict[str, str],
        criterion: Criterion,
        loss_weight: LossWeight | None = None,
    ) -> None:
        super().__init__(name=name, loss_weight=loss_weight)
        self.source_ref = coerce_ref(source_ref)
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
        source = context.resolve_tensor(self.source_ref)
        base_loss = self.criterion(source)
        return self._apply_postprocessing(base_loss, step=step)
