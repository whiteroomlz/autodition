import pytest
import torch

from src.data.components.batch import Sample
from src.data.components.collate import SchemaCollator
from src.data.components.schema import (
    CategoricalValueSpec,
    CustomBatchingSpec,
    FieldSpec,
    ListBatchingSpec,
    OpaqueValueSpec,
    PadBatchingSpec,
    ScalarShapeSpec,
    Schema,
    StackBatchingSpec,
    TensorShapeSpec,
    TensorValueSpec,
    TokenValueSpec,
    aligned_with,
)


def test_field_spec_validation_rejects_opaque_stack() -> None:
    with pytest.raises(
        ValueError,
        match="OpaqueValueSpec requires ListBatchingSpec or CustomBatchingSpec",
    ):
        FieldSpec(
            name="payload",
            role="meta",
            value=OpaqueValueSpec(),
            shape=ScalarShapeSpec(),
            batching=StackBatchingSpec(),
        )


def test_field_spec_validation_rejects_pad_without_variable_axis() -> None:
    with pytest.raises(ValueError, match="shape.variable_axes"):
        FieldSpec(
            name="mel",
            role="input",
            value=TensorValueSpec(dtype=torch.float32),
            shape=TensorShapeSpec(axes=("freq", "time"), variable_axes=("time",)),
            batching=PadBatchingSpec(variable_axis="freq", pad_value=0.0),
        )


def test_schema_validation_rejects_unknown_relation_target() -> None:
    with pytest.raises(ValueError, match="unknown related field"):
        Schema(
            fields={
                "tokens": FieldSpec(
                    name="tokens",
                    role="input",
                    value=TokenValueSpec(dtype=torch.long, vocab_size=128, pad_id=0),
                    shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                    batching=PadBatchingSpec(variable_axis="time", pad_value=0),
                    relations=(aligned_with("missing_mask", ("time",)),),
                )
            }
        )


def test_text_pretrain_collation_pads_tokens_and_attention_mask() -> None:
    schema = Schema(
        fields={
            "input_ids": FieldSpec(
                name="input_ids",
                role="input",
                value=TokenValueSpec(dtype=torch.long, vocab_size=1024, pad_id=0),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0),
            ),
            "attention_mask": FieldSpec(
                name="attention_mask",
                role="meta",
                value=TensorValueSpec(dtype=torch.bool),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=False),
                relations=(aligned_with("input_ids", ("time",)),),
            ),
        }
    )
    collator = SchemaCollator(schema=schema)

    batch = collator(
        [
            Sample("a", fields={"input_ids": [1, 2, 3], "attention_mask": [True, True, True]}),
            Sample("b", fields={"input_ids": [4, 5], "attention_mask": [True, True]}),
        ]
    )

    assert batch.fields["input_ids"].shape == (2, 3)
    assert batch.fields["attention_mask"].shape == (2, 3)
    assert batch.masks["input_ids"].tolist() == [[True, True, True], [True, True, False]]


def test_text_sft_collation_supports_loss_mask_field() -> None:
    schema = Schema(
        fields={
            "input_ids": FieldSpec(
                name="input_ids",
                role="input",
                value=TokenValueSpec(dtype=torch.long, vocab_size=4096, pad_id=0),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0),
            ),
            "loss_mask": FieldSpec(
                name="loss_mask",
                role="weight",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0.0),
                relations=(aligned_with("input_ids", ("time",)),),
            ),
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"input_ids": [1, 2, 3], "loss_mask": [0.0, 1.0, 1.0]}),
            Sample("b", fields={"input_ids": [4, 5], "loss_mask": [0.0, 1.0]}),
        ]
    )

    assert batch.fields["loss_mask"].shape == (2, 3)
    assert torch.allclose(batch.fields["loss_mask"][1], torch.tensor([0.0, 1.0, 0.0]))


def test_time_series_collation_supports_imputation_and_autoregression() -> None:
    schema = Schema(
        fields={
            "history": FieldSpec(
                name="history",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time", "feature"), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0.0),
            ),
            "future": FieldSpec(
                name="future",
                role="supervision",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time", "feature"), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0.0),
                relations=(aligned_with("history", ("feature",)),),
            ),
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"history": [[1.0], [2.0], [3.0]], "future": [[4.0], [5.0]]}),
            Sample("b", fields={"history": [[7.0], [8.0]], "future": [[9.0]]}),
        ]
    )

    assert batch.fields["history"].shape == (2, 3, 1)
    assert batch.fields["future"].shape == (2, 2, 1)
    assert batch.masks["history"].tolist() == [[True, True, True], [True, True, False]]


def test_image_change_detection_collation_stacks_paired_images() -> None:
    schema = Schema(
        fields={
            "image_a": FieldSpec(
                name="image_a",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("channel", "height", "width")),
                batching=StackBatchingSpec(),
            ),
            "image_b": FieldSpec(
                name="image_b",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("channel", "height", "width")),
                batching=StackBatchingSpec(),
            ),
            "change_label": FieldSpec(
                name="change_label",
                role="supervision",
                value=CategoricalValueSpec(dtype=torch.long, cardinality=2),
                shape=ScalarShapeSpec(),
                batching=StackBatchingSpec(),
            ),
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"image_a": torch.ones(3, 8, 8), "image_b": torch.zeros(3, 8, 8), "change_label": 0}),
            Sample("b", fields={"image_a": torch.zeros(3, 8, 8), "image_b": torch.ones(3, 8, 8), "change_label": 1}),
        ]
    )

    assert batch.fields["image_a"].shape == (2, 3, 8, 8)
    assert batch.fields["change_label"].tolist() == [0, 1]


def test_audio_classification_collation_keeps_waveform_list_and_stacks_labels() -> None:
    schema = Schema(
        fields={
            "waveform": FieldSpec(
                name="waveform",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=ListBatchingSpec(),
            ),
            "class_id": FieldSpec(
                name="class_id",
                role="supervision",
                value=CategoricalValueSpec(dtype=torch.long, cardinality=10),
                shape=ScalarShapeSpec(),
                batching=StackBatchingSpec(),
            ),
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"waveform": torch.tensor([1.0, 2.0, 3.0]), "class_id": 1}),
            Sample("b", fields={"waveform": torch.tensor([4.0, 5.0]), "class_id": 2}),
        ]
    )

    assert isinstance(batch.fields["waveform"], tuple)
    assert batch.fields["class_id"].shape == (2,)


def test_source_separation_collation_pads_dense_targets() -> None:
    schema = Schema(
        fields={
            "mixture_audio": FieldSpec(
                name="mixture_audio",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0.0),
            ),
            "sources_audio": FieldSpec(
                name="sources_audio",
                role="supervision",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("source", "time"), variable_axes=("time",)),
                batching=PadBatchingSpec(variable_axis="time", pad_value=0.0),
                relations=(aligned_with("mixture_audio", ("time",)),),
            ),
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"mixture_audio": [1.0, 2.0, 3.0], "sources_audio": [[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]}),
            Sample("b", fields={"mixture_audio": [4.0, 5.0], "sources_audio": [[4.0, 0.0], [0.0, 5.0]]}),
        ]
    )

    assert batch.fields["mixture_audio"].shape == (2, 3)
    assert batch.fields["sources_audio"].shape == (2, 2, 3)
    assert batch.masks["sources_audio"].tolist() == [[True, True, True], [True, True, False]]


def test_distillation_collation_supports_teacher_logits() -> None:
    schema = Schema(
        fields={
            "student_input": FieldSpec(
                name="student_input",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("feature",)),
                batching=StackBatchingSpec(),
            ),
            "teacher_logits": FieldSpec(
                name="teacher_logits",
                role="supervision",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("class",)),
                batching=StackBatchingSpec(),
            ),
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"student_input": [0.1, 0.2], "teacher_logits": [1.0, 2.0, 3.0]}),
            Sample("b", fields={"student_input": [0.3, 0.4], "teacher_logits": [3.0, 2.0, 1.0]}),
        ]
    )

    assert batch.fields["teacher_logits"].shape == (2, 3)


def test_clustering_collation_supports_input_only_batches() -> None:
    schema = Schema(
        fields={
            "embedding": FieldSpec(
                name="embedding",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("feature",)),
                batching=StackBatchingSpec(),
            )
        }
    )
    batch = SchemaCollator(schema)(
        [
            Sample("a", fields={"embedding": [1.0, 2.0, 3.0]}),
            Sample("b", fields={"embedding": [4.0, 5.0, 6.0]}),
        ]
    )

    assert batch.fields["embedding"].shape == (2, 3)


def test_custom_batching_handler_is_supported() -> None:
    schema = Schema(
        fields={
            "events": FieldSpec(
                name="events",
                role="meta",
                value=OpaqueValueSpec(),
                shape=ScalarShapeSpec(),
                batching=CustomBatchingSpec(handler_name="events"),
            )
        }
    )

    def events_handler(values, field_spec):
        return list(values), None

    batch = SchemaCollator(schema, custom_handlers={"events": events_handler})(
        [
            Sample("a", fields={"events": [{"t": 0.1}]}),
            Sample("b", fields={"events": [{"t": 0.2}, {"t": 0.3}]}),
        ]
    )

    assert isinstance(batch.fields["events"], list)
    assert len(batch.fields["events"]) == 2
