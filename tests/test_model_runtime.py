from __future__ import annotations

import hydra
import pytest
import torch

from src.data.components.batch import Batch
from src.models.components.base import (
    EncoderStage,
    HeadStage,
    ModelContext,
    PipelineModel,
    Ref,
    TensorSlot,
    TransformStage,
    setup_modules,
)
from src.models.components.composition.concat import ConcatFields
from src.models.components.metrics import MetricSuite
from src.models.components.objectives import (
    CrossEntropyCriterion,
    MeanSquaredErrorCriterion,
    ObjectiveComposer,
    SupervisedLossTerm,
)
from src.models.task_module import TaskModule


class CopyField(EncoderStage):
    def __init__(self, input_field: str, output_name: str) -> None:
        super().__init__(inputs=[Ref(source="field", name=input_field)], outputs=(output_name,))

    def forward(self, context: ModelContext) -> ModelContext:
        slot = context.resolve_slot(self.inputs[0])
        context.write("rep", self.outputs[0], TensorSlot(value=slot.value.float(), mask=slot.mask))
        return context


class LinearHead(HeadStage):
    def __init__(self, input_ref: Ref, output_name: str, in_dim: int, out_dim: int) -> None:
        super().__init__(inputs=[input_ref], outputs=(output_name,))
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, context: ModelContext) -> ModelContext:
        slot = context.resolve_slot(self.inputs[0])
        context.write(self.store, self.outputs[0], TensorSlot(value=self.linear(slot.value), mask=slot.mask))
        return context


class FakeSeparatorHead(HeadStage):
    def __init__(self, input_ref: Ref, output_name: str) -> None:
        super().__init__(inputs=[input_ref], outputs=(output_name,))

    def forward(self, context: ModelContext) -> ModelContext:
        slot = context.resolve_slot(self.inputs[0])
        waveform = slot.value
        separated = torch.stack((waveform, torch.zeros_like(waveform)), dim=1)
        context.write(self.store, self.outputs[0], TensorSlot(value=separated, mask=slot.mask))
        return context


class SourceReencoder(TransformStage):
    def __init__(self, input_ref: Ref, output_name: str) -> None:
        super().__init__(inputs=[input_ref], outputs=(output_name,))

    def forward(self, context: ModelContext) -> ModelContext:
        slot = context.resolve_slot(self.inputs[0])
        hidden_state = slot.value.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        context.write(self.store, self.outputs[0], TensorSlot(value=hidden_state))
        return context


def _optimizer_factory():
    return hydra.utils.get_object("torch.optim.AdamW")


def test_pipeline_model_rejects_unresolved_rep_dependency() -> None:
    model = PipelineModel(
        stages=[LinearHead(Ref(source="rep", name="missing"), "class_logits", in_dim=4, out_dim=3)]
    )

    with pytest.raises(ValueError, match="unavailable rep"):
        setup_modules(model)


def test_pipeline_model_rejects_duplicate_pred_writer() -> None:
    model = PipelineModel(
        stages=[
            CopyField("features", "shared"),
            LinearHead(Ref(source="rep", name="shared"), "class_logits", in_dim=4, out_dim=3),
            LinearHead(Ref(source="rep", name="shared"), "class_logits", in_dim=4, out_dim=3),
        ]
    )

    with pytest.raises(ValueError, match="already has a producer"):
        setup_modules(model)


def test_task_module_supports_classification_and_regression() -> None:
    model = PipelineModel(
        stages=[
            CopyField("features", "shared"),
            LinearHead(Ref(source="rep", name="shared"), "class_logits", in_dim=4, out_dim=3),
            LinearHead(Ref(source="rep", name="shared"), "score", in_dim=4, out_dim=1),
        ]
    )
    module = TaskModule(
        model=model,
        objectives=ObjectiveComposer(
            terms=[
                SupervisedLossTerm(
                    name="classification",
                    prediction_ref=Ref(source="pred", name="class_logits"),
                    target_ref=Ref(source="field", name="class_id"),
                    criterion=CrossEntropyCriterion(),
                ),
                SupervisedLossTerm(
                    name="regression",
                    prediction_ref=Ref(source="pred", name="score"),
                    target_ref=Ref(source="field", name="score_target"),
                    criterion=MeanSquaredErrorCriterion(),
                ),
            ]
        ),
        metrics=MetricSuite(terms=[]),
        optimizer=_optimizer_factory(),
        scheduler=None,
        compile=False,
        monitor_metric="val/loss",
        monitor_metric_mode="min",
    )
    module.setup("fit")

    batch = Batch(
        sample_ids=("a", "b"),
        fields={
            "features": torch.randn(2, 4),
            "class_id": torch.tensor([0, 2]),
            "score_target": torch.randn(2, 1),
        },
        masks={},
        meta={},
    )
    loss, result, term_losses = module.model_step(batch, split="train")

    assert loss.ndim == 0
    assert set(result.preds) == {"class_logits", "score"}
    assert set(term_losses) == {"classification", "regression"}


def test_task_module_supports_classification_and_reconstruction() -> None:
    model = PipelineModel(
        stages=[
            CopyField("embedding", "shared"),
            LinearHead(Ref(source="rep", name="shared"), "class_logits", in_dim=4, out_dim=3),
            LinearHead(Ref(source="rep", name="shared"), "reconstruction", in_dim=4, out_dim=4),
        ]
    )
    module = TaskModule(
        model=model,
        objectives=ObjectiveComposer(
            terms=[
                SupervisedLossTerm(
                    name="classification",
                    prediction_ref=Ref(source="pred", name="class_logits"),
                    target_ref=Ref(source="field", name="class_id"),
                    criterion=CrossEntropyCriterion(),
                ),
                SupervisedLossTerm(
                    name="reconstruction",
                    prediction_ref=Ref(source="pred", name="reconstruction"),
                    target_ref=Ref(source="field", name="embedding_target"),
                    criterion=MeanSquaredErrorCriterion(),
                ),
            ]
        ),
        metrics=MetricSuite(terms=[]),
        optimizer=_optimizer_factory(),
        scheduler=None,
        compile=False,
        monitor_metric="val/loss",
        monitor_metric_mode="min",
    )
    module.setup("fit")

    batch = Batch(
        sample_ids=("a", "b"),
        fields={
            "embedding": torch.randn(2, 4),
            "class_id": torch.tensor([1, 0]),
            "embedding_target": torch.randn(2, 4),
        },
        masks={},
        meta={},
    )
    loss, result, _ = module.model_step(batch, split="train")

    assert loss.ndim == 0
    assert result.preds["reconstruction"].value.shape == (2, 4)


def test_pipeline_model_supports_two_input_fusion() -> None:
    model = PipelineModel(
        stages=[
            ConcatFields(
                inputs=[
                    Ref(source="field", name="audio_embedding"),
                    Ref(source="field", name="tabular_embedding"),
                ],
                output_name="fused",
            ),
            LinearHead(Ref(source="rep", name="fused"), "class_logits", in_dim=6, out_dim=3),
        ]
    )
    setup_modules(model)

    batch = Batch(
        sample_ids=("a", "b"),
        fields={
            "audio_embedding": torch.randn(2, 4),
            "tabular_embedding": torch.randn(2, 2),
        },
        masks={},
        meta={},
    )
    result = model(batch)

    assert result.preds["class_logits"].value.shape == (2, 3)


def test_task_module_supports_dense_source_separation_supervision() -> None:
    model = PipelineModel(
        stages=[
            FakeSeparatorHead(Ref(source="field", name="mixture_audio"), "sources_audio"),
        ]
    )
    module = TaskModule(
        model=model,
        objectives=ObjectiveComposer(
            terms=[
                SupervisedLossTerm(
                    name="separation",
                    prediction_ref=Ref(source="pred", name="sources_audio"),
                    target_ref=Ref(source="field", name="sources_audio"),
                    criterion=MeanSquaredErrorCriterion(),
                    mask_ref=Ref(source="field", name="sources_audio"),
                )
            ]
        ),
        metrics=MetricSuite(terms=[]),
        optimizer=_optimizer_factory(),
        scheduler=None,
        compile=False,
        monitor_metric="val/loss",
        monitor_metric_mode="min",
    )
    module.setup("fit")

    batch = Batch(
        sample_ids=("a", "b"),
        fields={
            "mixture_audio": torch.randn(2, 5),
            "sources_audio": torch.randn(2, 2, 5),
        },
        masks={
            "mixture_audio": torch.tensor([[True, True, True, True, True], [True, True, True, False, False]]),
            "sources_audio": torch.tensor([[True, True, True, True, True], [True, True, True, False, False]]),
        },
        meta={},
    )
    loss, result, _ = module.model_step(batch, split="train")

    assert loss.ndim == 0
    assert result.preds["sources_audio"].value.shape == (2, 2, 5)


def test_pipeline_model_supports_composite_audio_pipeline() -> None:
    model = PipelineModel(
        stages=[
            CopyField("waveform", "audio_shared"),
            LinearHead(Ref(source="rep", name="audio_shared"), "signal_logits", in_dim=8, out_dim=2),
            FakeSeparatorHead(Ref(source="field", name="waveform"), "sources_audio"),
            SourceReencoder(Ref(source="pred", name="sources_audio"), "source_features"),
            LinearHead(Ref(source="rep", name="source_features"), "class_logits", in_dim=1, out_dim=3),
            LinearHead(Ref(source="rep", name="source_features"), "doa", in_dim=1, out_dim=1),
        ]
    )
    setup_modules(model)

    batch = Batch(
        sample_ids=("a", "b"),
        fields={"waveform": torch.randn(2, 8)},
        masks={"waveform": torch.ones(2, 8, dtype=torch.bool)},
        meta={},
    )
    result = model(batch)

    assert set(result.preds) == {"signal_logits", "sources_audio", "class_logits", "doa"}
