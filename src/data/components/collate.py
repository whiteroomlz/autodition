from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .batch import Batch, Sample
from .schema import (
    CategoricalValueSpec,
    CustomBatchingSpec,
    FieldSpec,
    ListBatchingSpec,
    PadBatchingSpec,
    ReferenceValueSpec,
    Schema,
    StackBatchingSpec,
    TensorShapeSpec,
    TensorValueSpec,
    TokenValueSpec,
)

BatchHandler = Callable[[Sequence[Any], FieldSpec], Tuple[Any, Optional[torch.BoolTensor]] | Any]


class Collator(ABC):
    def __init__(self, schema: Schema, custom_handlers: Optional[Dict[str, BatchHandler]] = None):
        self.schema = schema
        self.custom_handlers = custom_handlers or {}

    @abstractmethod
    def __call__(self, samples: List[Sample]) -> Batch:
        raise NotImplementedError


class SchemaCollator(Collator):
    def __call__(self, samples: List[Sample]) -> Batch:
        if not samples:
            raise ValueError("SchemaCollator expects at least one sample")

        sample_ids = tuple(sample.sample_id for sample in samples)
        fields: Dict[str, Any] = {}
        masks: Dict[str, torch.BoolTensor] = {}
        sample_meta = tuple(sample.meta for sample in samples)

        for field_name, field_spec in self.schema.fields.items():
            values = [sample.fields.get(field_name) for sample in samples]

            if all(value is None for value in values):
                if field_spec.required:
                    raise KeyError(f"Required field '{field_name}' is missing from all samples")
                continue

            if any(value is None for value in values):
                raise ValueError(f"Field '{field_name}' is partially missing inside the batch")

            collated_value, field_mask = self._collate_field(field_spec, values)
            fields[field_name] = collated_value
            if field_mask is not None:
                masks[field_name] = field_mask

        return Batch(
            sample_ids=sample_ids,
            fields=fields,
            masks=masks,
            meta={"sample_meta": sample_meta},
        )

    def _collate_field(
        self,
        field_spec: FieldSpec,
        values: Sequence[Any],
    ) -> Tuple[Any, Optional[torch.BoolTensor]]:
        batching = field_spec.batching

        if isinstance(batching, StackBatchingSpec):
            return self._stack_values(field_spec, values), None

        if isinstance(batching, PadBatchingSpec):
            return self._pad_values(field_spec, values)

        if isinstance(batching, ListBatchingSpec):
            return tuple(values), None

        if isinstance(batching, CustomBatchingSpec):
            return self._custom_batch(field_spec, values)

        raise TypeError(f"Unsupported batching spec for field '{field_spec.name}'")

    def _stack_values(self, field_spec: FieldSpec, values: Sequence[Any]) -> torch.Tensor:
        dtype = self._value_dtype(field_spec)
        tensors = [torch.as_tensor(value, dtype=dtype) for value in values]
        reference_shape = tensors[0].shape
        if any(tensor.shape != reference_shape for tensor in tensors[1:]):
            raise ValueError(f"Field '{field_spec.name}' tensors must have identical shapes")
        return torch.stack(tensors, dim=0)

    def _pad_values(
        self,
        field_spec: FieldSpec,
        values: Sequence[Any],
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        if not isinstance(field_spec.shape, TensorShapeSpec):
            raise TypeError("PadBatchingSpec requires TensorShapeSpec")

        dtype = self._value_dtype(field_spec)
        tensors = [torch.as_tensor(value, dtype=dtype) for value in values]
        if any(tensor.ndim != tensors[0].ndim for tensor in tensors[1:]):
            raise ValueError(f"Field '{field_spec.name}' tensors must have matching ranks")

        axis_index = field_spec.shape.axes.index(field_spec.batching.variable_axis)
        reference_shape = tensors[0].shape
        for tensor in tensors[1:]:
            for dim_index, (base_dim, current_dim) in enumerate(zip(reference_shape, tensor.shape)):
                if dim_index != axis_index and base_dim != current_dim:
                    raise ValueError(
                        f"Field '{field_spec.name}' tensors must match on all axes except "
                        f"'{field_spec.batching.variable_axis}'"
                    )

        lengths = torch.tensor([tensor.shape[axis_index] for tensor in tensors], dtype=torch.long)
        max_length = int(lengths.max().item())

        padded_shape = [len(tensors), *reference_shape]
        padded_shape[axis_index + 1] = max_length
        padded = torch.full(
            tuple(padded_shape),
            fill_value=field_spec.batching.pad_value,
            dtype=dtype,
            device=tensors[0].device,
        )

        for sample_index, tensor in enumerate(tensors):
            target = padded[sample_index]
            slices = [slice(None)] * target.ndim
            slices[axis_index] = slice(0, tensor.shape[axis_index])
            target[tuple(slices)] = tensor

        mask = torch.arange(max_length, device=padded.device)[None, :] < lengths[:, None]
        return padded, mask

    def _custom_batch(
        self,
        field_spec: FieldSpec,
        values: Sequence[Any],
    ) -> Tuple[Any, Optional[torch.BoolTensor]]:
        handler = self.custom_handlers.get(field_spec.batching.handler_name)
        if handler is None:
            raise KeyError(
                f"No custom batching handler registered for '{field_spec.batching.handler_name}'"
            )

        result = handler(values, field_spec)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, None

    @staticmethod
    def _value_dtype(field_spec: FieldSpec) -> torch.dtype:
        value_spec = field_spec.value
        if isinstance(value_spec, (TensorValueSpec, CategoricalValueSpec, TokenValueSpec)):
            return value_spec.dtype
        if isinstance(value_spec, ReferenceValueSpec):
            raise TypeError(
                f"Field '{field_spec.name}' with value spec {type(value_spec).__name__} "
                "cannot be tensor-collated"
            )
        raise TypeError(f"Unsupported value spec for field '{field_spec.name}'")
