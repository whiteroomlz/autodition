"""Metric orchestration and shared helpers."""

from __future__ import annotations

import copy
from typing import Dict, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric

from src.data.components.batch import Batch

from ..base import ModelResult, Ref


class MetricTerm(torch.nn.Module):
    """Single named metric updater bound to one or more refs."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def update(self, batch: Batch, result: ModelResult) -> None:
        raise NotImplementedError

    def metric_objects(self) -> Dict[str, Metric]:
        raise NotImplementedError

    def clone(self) -> MetricTerm:
        return copy.deepcopy(self)


class MetricSuite(torch.nn.Module):
    """Collection of metric terms updated together during a loop stage."""

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


def apply_mask(
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


def apply_prediction_mask(prediction: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    if prediction.shape == mask.shape:
        return prediction[mask]
    if prediction.ndim == mask.ndim + 1 and prediction.shape[:-1] == mask.shape:
        return prediction[mask]
    raise ValueError("Mask shape is incompatible with prediction tensor")
