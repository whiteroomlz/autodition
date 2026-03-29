"""Source-separation-specific metric terms."""

from __future__ import annotations

from typing import Dict

import torch
from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric

from src.data.components.batch import Batch

from ..base import ModelContext, ModelResult, Ref
from ..separation import (
    align_sources,
    build_best_permutation,
    compute_si_sdr,
    flatten_active_sources,
    infer_source_activity,
    reduce_loss_over_nonbatch_dims,
    sum_sources,
)
from .base import MetricTerm, coerce_optional_ref, coerce_ref


class PermutationInvariantSISDRMetricTerm(MetricTerm):
    """SI-SDR over separation outputs after best-permutation alignment."""

    def __init__(
        self,
        name: str,
        prediction_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        activity_ref: Ref | Dict[str, str] | None = None,
        baseline_ref: Ref | Dict[str, str] | None = None,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.prediction_ref = coerce_ref(prediction_ref)
        self.target_ref = coerce_ref(target_ref)
        self.activity_ref = coerce_optional_ref(activity_ref)
        self.baseline_ref = coerce_optional_ref(baseline_ref)
        self.mask_ref = coerce_optional_ref(mask_ref)
        self.metric = MeanMetric()

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        prediction = context.resolve_tensor(self.prediction_ref)
        target = context.resolve_tensor(self.target_ref)
        mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        activity = (
            context.resolve_tensor(self.activity_ref).bool()
            if self.activity_ref is not None
            else infer_source_activity(target, mask=mask)
        )

        pairwise_cost = self._pairwise_negative_sisdr(prediction, target, mask)
        permutation, _ = build_best_permutation(pairwise_cost)
        aligned_prediction = align_sources(prediction, permutation)

        active_prediction, active_target, active_mask = flatten_active_sources(
            aligned_prediction,
            target,
            activity,
            mask,
        )
        if active_prediction.numel() == 0:
            self.metric.update(torch.zeros(1, device=prediction.device, dtype=prediction.dtype))
            return

        si_sdr = compute_si_sdr(active_prediction, active_target, mask=active_mask)
        if self.baseline_ref is not None:
            baseline = context.resolve_tensor(self.baseline_ref)
            expanded_baseline = baseline.unsqueeze(1).expand_as(target)
            active_baseline, _, _ = flatten_active_sources(
                expanded_baseline,
                target,
                activity,
                mask,
            )
            si_sdr = si_sdr - compute_si_sdr(active_baseline, active_target, mask=active_mask)

        self.metric.update(si_sdr)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}

    @staticmethod
    def _pairwise_negative_sisdr(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.BoolTensor | None,
    ) -> torch.Tensor:
        pairwise_cost = []
        for prediction_index in range(prediction.shape[1]):
            row = []
            for target_index in range(target.shape[1]):
                row.append(
                    -compute_si_sdr(
                        prediction[:, prediction_index, :],
                        target[:, target_index, :],
                        mask=mask,
                    )
                )
            pairwise_cost.append(torch.stack(row, dim=1))
        return torch.stack(pairwise_cost, dim=1)


class SummedSourcesMetricTerm(MetricTerm):
    """Metric over the sum of separated sources against the mixture."""

    def __init__(
        self,
        name: str,
        sources_ref: Ref | Dict[str, str],
        target_ref: Ref | Dict[str, str],
        metric: Metric,
        mask_ref: Ref | Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.sources_ref = coerce_ref(sources_ref)
        self.target_ref = coerce_ref(target_ref)
        self.mask_ref = coerce_optional_ref(mask_ref)
        self.metric = metric

    def update(self, batch: Batch, result: ModelResult) -> None:
        context = ModelContext.from_batch(batch)
        context.reps.update(result.reps)
        context.preds.update(result.preds)

        sources = context.resolve_tensor(self.sources_ref)
        target = context.resolve_tensor(self.target_ref)
        mask = context.resolve_mask(self.mask_ref) if self.mask_ref is not None else None
        summed_sources = sum_sources(sources)

        if mask is not None:
            loss = reduce_loss_over_nonbatch_dims((summed_sources - target).abs(), mask=mask)
            self.metric.update(loss)
            return

        self.metric(summed_sources, target)

    def metric_objects(self) -> Dict[str, Metric]:
        return {self.name: self.metric}
