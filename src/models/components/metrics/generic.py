"""Generic supervised and pairwise metrics."""

from __future__ import annotations

from typing import Dict

import torch
from torchmetrics import Metric

from src.data.components.batch import Batch

from ..base import ModelContext, ModelResult, Ref
from .base import (
    MetricTerm,
    apply_mask,
    apply_prediction_mask,
    coerce_optional_ref,
    coerce_ref,
)


class SupervisedMetricTerm(MetricTerm):
    """Metric comparing a prediction ref against a supervision ref."""

    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.prediction_ref = coerce_ref(prediction_ref)
        self.target_ref = coerce_ref(target_ref)
        self.metric = metric
        self.mask_ref = coerce_optional_ref(mask_ref)

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        target = context.resolve_tensor(self.target_ref)
        if self.mask_ref is not None:
            mask = context.resolve_mask(self.mask_ref)
            prediction, target = apply_mask(prediction, target, mask)

        if target.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            self.metric(prediction, target.long())
        else:
            self.metric(prediction, target)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}


class PairMetricTerm(MetricTerm):
    """Metric over two arbitrary refs, useful for consistency or reconstruction."""

    def __init__(
        self,
        name: str,
        left_ref: Ref | Dict[str, str],
        right_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.left_ref = coerce_ref(left_ref)
        self.right_ref = coerce_ref(right_ref)
        self.metric = metric
        self.mask_ref = coerce_optional_ref(mask_ref)

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        left = context.resolve_tensor(self.left_ref)
        right = context.resolve_tensor(self.right_ref)
        if self.mask_ref is not None:
            mask = context.resolve_mask(self.mask_ref)
            left, right = apply_mask(left, right, mask)

        self.metric(left, right)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}


class PredictionOnlyMetricTerm(MetricTerm):
    """Metric consuming only a prediction ref, such as confidence diagnostics."""

    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.prediction_ref = coerce_ref(prediction_ref)
        self.metric = metric
        self.mask_ref = coerce_optional_ref(mask_ref)

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        if self.mask_ref is not None:
            mask = context.resolve_mask(self.mask_ref)
            prediction = apply_prediction_mask(prediction, mask)

        self.metric(prediction)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}
