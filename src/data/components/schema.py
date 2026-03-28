"""Strict field-based schema contracts for dataset samples and batches."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence, Tuple

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

FieldRole = Literal["input", "supervision", "meta", "weight", "id"]
ReferenceKind = Literal["path", "uri", "key", "blob"]

FIELD_ROLES = frozenset({"input", "supervision", "meta", "weight", "id"})
REFERENCE_KINDS = frozenset({"path", "uri", "key", "blob"})


class ValueSpec(ABC):
    """Describe what kind of value a field carries, independent of shape."""


@dataclass(frozen=True)
class TensorValueSpec(ValueSpec):
    """Dense tensor payload such as audio, images, masks, or embeddings."""

    dtype: torch.dtype


@dataclass(frozen=True)
class CategoricalValueSpec(ValueSpec):
    """Integer-coded categorical target or feature with fixed cardinality."""

    dtype: torch.dtype
    cardinality: int
    labels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        if self.cardinality <= 0:
            raise ValueError("CategoricalValueSpec.cardinality must be positive")
        if self.labels and len(self.labels) != self.cardinality:
            raise ValueError("CategoricalValueSpec.labels must match cardinality")


@dataclass(frozen=True)
class TokenValueSpec(ValueSpec):
    """Tokenized discrete sequence with explicit vocabulary and special ids."""

    dtype: torch.dtype
    vocab_size: int
    pad_id: int
    bos_id: int | None = None
    eos_id: int | None = None
    unk_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("TokenValueSpec.vocab_size must be positive")


@dataclass(frozen=True)
class ReferenceValueSpec(ValueSpec):
    """External reference such as a path or blob key resolved outside the model."""

    ref_kind: ReferenceKind

    def __post_init__(self) -> None:
        if self.ref_kind not in REFERENCE_KINDS:
            raise ValueError(f"Unsupported reference kind: {self.ref_kind}")


@dataclass(frozen=True)
class OpaqueValueSpec(ValueSpec):
    """Structured payload that stays outside dense tensor semantics."""

    pass


class ShapeSpec(ABC):
    """Describe per-sample structure independently from value semantics."""


@dataclass(frozen=True)
class ScalarShapeSpec(ShapeSpec):
    """Scalar value with no named axes."""

    pass


@dataclass(frozen=True)
class TensorShapeSpec(ShapeSpec):
    """Named tensor axes plus an optional subset of variable-length axes."""

    axes: Tuple[str, ...]
    variable_axes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        variable_axes = tuple(self.variable_axes)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "variable_axes", variable_axes)

        if not axes:
            raise ValueError("TensorShapeSpec.axes must not be empty")
        if len(axes) != len(set(axes)):
            raise ValueError("TensorShapeSpec.axes must be unique")
        if not set(variable_axes).issubset(set(axes)):
            raise ValueError("TensorShapeSpec.variable_axes must be a subset of axes")


@dataclass(frozen=True)
class OpaqueShapeSpec(ShapeSpec):
    """Non-tensor structure whose shape is intentionally left unspecified."""

    pass


class BatchingSpec(ABC):
    """Describe how collators combine field values across samples."""


@dataclass(frozen=True)
class StackBatchingSpec(BatchingSpec):
    """Require identical per-sample shape and stack values into one tensor."""

    pass


@dataclass(frozen=True)
class PadBatchingSpec(BatchingSpec):
    """Pad a declared variable axis and emit a mask for the padded dimension."""

    variable_axis: str
    pad_value: Any = 0


@dataclass(frozen=True)
class ListBatchingSpec(BatchingSpec):
    """Keep values as a Python list when dense collation is inappropriate."""

    pass


@dataclass(frozen=True)
class CustomBatchingSpec(BatchingSpec):
    """Delegate collation to a named custom handler registered in the collator."""

    handler_name: str

    def __post_init__(self) -> None:
        if not self.handler_name:
            raise ValueError("CustomBatchingSpec.handler_name must not be empty")


class RelationSpec(ABC):
    """Relationship between fields used for validation and downstream intent."""


@dataclass(frozen=True)
class PairedWith(RelationSpec):
    field_name: str


@dataclass(frozen=True)
class AlignedWith(RelationSpec):
    field_name: str
    axes: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))


@dataclass(frozen=True)
class DerivedFrom(RelationSpec):
    field_name: str


def paired_with(field_name: str) -> PairedWith:
    return PairedWith(field_name=field_name)


def aligned_with(field_name: str, axes: Sequence[str]) -> AlignedWith:
    return AlignedWith(field_name=field_name, axes=tuple(axes))


def derived_from(field_name: str) -> DerivedFrom:
    return DerivedFrom(field_name=field_name)


@dataclass(frozen=True)
class FieldSpec:
    """Single public schema atom describing one named field end to end."""

    name: str
    role: FieldRole
    value: ValueSpec
    shape: ShapeSpec
    batching: BatchingSpec
    required: bool = True
    relations: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _to_object(self.value))
        object.__setattr__(self, "shape", _to_object(self.shape))
        object.__setattr__(self, "batching", _to_object(self.batching))

        relations = tuple(_to_object(relation) for relation in self.relations)
        object.__setattr__(self, "relations", relations)
        self._validate()

    def _validate(self) -> None:
        if not self.name:
            raise ValueError("FieldSpec.name must not be empty")
        if self.role not in FIELD_ROLES:
            raise ValueError(f"Unsupported field role: {self.role}")

        if isinstance(self.value, OpaqueValueSpec) and not isinstance(
            self.batching, (ListBatchingSpec, CustomBatchingSpec)
        ):
            raise ValueError("OpaqueValueSpec requires ListBatchingSpec or CustomBatchingSpec")

        if isinstance(self.value, ReferenceValueSpec) and not isinstance(
            self.batching, (ListBatchingSpec, CustomBatchingSpec)
        ):
            raise ValueError("ReferenceValueSpec requires ListBatchingSpec or CustomBatchingSpec")

        if isinstance(self.shape, OpaqueShapeSpec) and not isinstance(
            self.batching, (ListBatchingSpec, CustomBatchingSpec)
        ):
            raise ValueError("OpaqueShapeSpec requires ListBatchingSpec or CustomBatchingSpec")

        if isinstance(self.batching, PadBatchingSpec):
            if not isinstance(self.shape, TensorShapeSpec):
                raise ValueError("PadBatchingSpec requires TensorShapeSpec")
            if self.batching.variable_axis not in self.shape.axes:
                raise ValueError("PadBatchingSpec.variable_axis must be declared in shape.axes")
            if self.batching.variable_axis not in self.shape.variable_axes:
                raise ValueError(
                    "PadBatchingSpec.variable_axis must be declared in shape.variable_axes"
                )

        if isinstance(self.batching, StackBatchingSpec):
            if isinstance(self.shape, TensorShapeSpec) and self.shape.variable_axes:
                raise ValueError("StackBatchingSpec requires fixed-size TensorShapeSpec")
            if isinstance(self.value, (OpaqueValueSpec, ReferenceValueSpec)):
                raise ValueError("StackBatchingSpec is not supported for opaque/reference values")

        if isinstance(self.shape, ScalarShapeSpec) and isinstance(self.batching, PadBatchingSpec):
            raise ValueError("ScalarShapeSpec cannot use PadBatchingSpec")

        if isinstance(self.value, TokenValueSpec) and not isinstance(self.shape, TensorShapeSpec):
            raise ValueError("TokenValueSpec requires TensorShapeSpec")


@dataclass(frozen=True)
class Schema:
    """Source of truth for field contracts consumed by datasets and collators."""

    fields: Dict[str, FieldSpec]

    def __post_init__(self) -> None:
        fields = {
            field_name: _to_object(field_spec)
            for field_name, field_spec in dict(self.fields).items()
        }
        object.__setattr__(self, "fields", fields)

        if not fields:
            raise ValueError("Schema.fields must not be empty")

        for field_name, field_spec in fields.items():
            if field_name != field_spec.name:
                raise ValueError(
                    f"Schema key '{field_name}' does not match FieldSpec.name '{field_spec.name}'"
                )

        self._validate_relations()

    def _validate_relations(self) -> None:
        for field_name, field_spec in self.fields.items():
            for relation in field_spec.relations:
                related_field_name = relation.field_name
                if related_field_name not in self.fields:
                    raise ValueError(
                        f"Field '{field_name}' references unknown related field '{related_field_name}'"
                    )

                if isinstance(relation, AlignedWith):
                    current_shape = field_spec.shape
                    related_shape = self.fields[related_field_name].shape
                    if not isinstance(current_shape, TensorShapeSpec) or not isinstance(
                        related_shape, TensorShapeSpec
                    ):
                        raise ValueError("AlignedWith requires tensor-shaped fields")
                    if not set(relation.axes).issubset(set(current_shape.axes)):
                        raise ValueError(
                            f"Field '{field_name}' alignment axes must exist on the source field"
                        )
                    if not set(relation.axes).issubset(set(related_shape.axes)):
                        raise ValueError(
                            f"Field '{field_name}' alignment axes must exist on the related field"
                        )

    def __contains__(self, field_name: str) -> bool:
        return field_name in self.fields

    def __getitem__(self, field_name: str) -> FieldSpec:
        return self.fields[field_name]

    def require_field(self, field_name: str) -> FieldSpec:
        if field_name not in self.fields:
            raise KeyError(f"Schema does not contain field '{field_name}'")
        return self.fields[field_name]

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(self.fields.keys())

    def field_names_by_role(self, role: FieldRole) -> Tuple[str, ...]:
        return tuple(
            field_name
            for field_name, field_spec in self.fields.items()
            if field_spec.role == role
        )

    def fields_by_role(self, role: FieldRole) -> Tuple[FieldSpec, ...]:
        return tuple(
            field_spec for field_spec in self.fields.values() if field_spec.role == role
        )

    @property
    def input_field_names(self) -> Tuple[str, ...]:
        return self.field_names_by_role("input")

    @property
    def supervision_field_names(self) -> Tuple[str, ...]:
        return self.field_names_by_role("supervision")

    @property
    def meta_field_names(self) -> Tuple[str, ...]:
        return self.field_names_by_role("meta")

    @property
    def weight_field_names(self) -> Tuple[str, ...]:
        return self.field_names_by_role("weight")

    @property
    def id_field_names(self) -> Tuple[str, ...]:
        return self.field_names_by_role("id")

    def categorical_supervision_field_names(self) -> Tuple[str, ...]:
        return tuple(
            field_name
            for field_name in self.supervision_field_names
            if isinstance(self.fields[field_name].value, CategoricalValueSpec)
        )


def _to_object(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_object(value)
    return value
