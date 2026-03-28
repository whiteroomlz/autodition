from __future__ import annotations

import copy
from typing import Dict, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric

from src.data.components.batch import Batch

from .base import ModelContext, ModelResult, Ref


class MetricTerm(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def update(self, batch: Batch, result: ModelResult) -> None:
        raise NotImplementedError

    def metric_objects(self) -> Dict[str, Metric]:
        raise NotImplementedError

    def clone(self) -> MetricTerm:
        return copy.deepcopy(self)


class SupervisedMetricTerm(MetricTerm):
    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.prediction_ref = _coerce_ref(prediction_ref)
        self.target_ref = _coerce_ref(target_ref)
        self.metric = metric
        self.mask_ref = _coerce_optional_ref(mask_ref)

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        target = context.resolve_tensor(self.target_ref)

        if self.mask_ref is not None:
            mask = context.resolve_mask(self.mask_ref)
            prediction, target = _apply_mask(prediction, target, mask)

        if target.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            self.metric(prediction, target.long())
        else:
            self.metric(prediction, target)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}


class PairMetricTerm(MetricTerm):
    def __init__(
        self,
        name: str,
        left_ref: Ref | Dict[str, str],
        right_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.left_ref = _coerce_ref(left_ref)
        self.right_ref = _coerce_ref(right_ref)
        self.metric = metric
        self.mask_ref = _coerce_optional_ref(mask_ref)

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        left = context.resolve_tensor(self.left_ref)
        right = context.resolve_tensor(self.right_ref)

        if self.mask_ref is not None:
            mask = context.resolve_mask(self.mask_ref)
            left, right = _apply_mask(left, right, mask)

        self.metric(left, right)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}


class PredictionOnlyMetricTerm(MetricTerm):
    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.prediction_ref = _coerce_ref(prediction_ref)
        self.metric = metric
        self.mask_ref = _coerce_optional_ref(mask_ref)

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        if self.mask_ref is not None:
            mask = context.resolve_mask(self.mask_ref)
            prediction = _apply_prediction_mask(prediction, mask)

        self.metric(prediction)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}


class MetricSuite(torch.nn.Module):
    def __init__(self, terms: Sequence[MetricTerm]) -> None:
        super().__init__()
        self.terms = torch.nn.ModuleList(terms)

    def update(self, batch: Batch, result: ModelResult) -> None:
        for term in self.terms:
            term.update(batch, result)

    def metric_objects(self) -> Dict[str, Metric]:
        metrics: Dict[str, Metric] = {}
        for term in self.terms:
            metrics.update(term.metric_objects())
        return metrics

    def clone(self) -> MetricSuite:
        return copy.deepcopy(self)


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


def _apply_mask(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape == target.shape:
        return prediction[mask], target[mask]

    if prediction.ndim == target.ndim + 1 and prediction.shape[:-1] == target.shape:
        return prediction[mask], target[mask]

    raise ValueError(
        "Mask-based metric filtering requires equal prediction/target shapes or "
        "classification-style logits with one extra class dimension"
    )


def _apply_prediction_mask(prediction: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    if prediction.shape == mask.shape:
        return prediction[mask]
    if prediction.ndim == mask.ndim + 1 and prediction.shape[:-1] == mask.shape:
        return prediction[mask]
    raise ValueError("Mask shape is incompatible with prediction tensor")
