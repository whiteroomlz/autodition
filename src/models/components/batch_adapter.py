from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.data.components.batch import Batch

from .base import ModelInput, SequentialModelInput


@dataclass(frozen=True)
class BatchToModelInputAdapter:
    raw_field: Optional[str] = None
    numerical_field: Optional[str] = None
    categorical_field: Optional[str] = None
    padding_mask_field: Optional[str] = None

    def __call__(self, batch: Batch) -> ModelInput:
        kwargs: dict[str, Any] = {}

        if self.raw_field is not None:
            kwargs["raw"] = self._get_batch_field(batch, self.raw_field, tupleify=True)

        if self.numerical_field is not None:
            kwargs["numerical"] = self._get_batch_field(batch, self.numerical_field)

        if self.categorical_field is not None:
            kwargs["categorical"] = self._get_batch_field(batch, self.categorical_field)

        if self.padding_mask_field is not None:
            if self.padding_mask_field not in batch.masks:
                raise KeyError(
                    f"Batch does not contain a padding mask for field '{self.padding_mask_field}'"
                )
            kwargs["padding_mask"] = batch.masks[self.padding_mask_field]

        if "padding_mask" in kwargs:
            return SequentialModelInput(**kwargs)

        return ModelInput(**kwargs)

    @staticmethod
    def _get_batch_field(batch: Batch, field_name: str, tupleify: bool = False) -> Any:
        if field_name not in batch.fields:
            raise KeyError(f"Batch does not contain field '{field_name}'")

        value = batch.fields[field_name]
        if tupleify and not isinstance(value, tuple):
            if isinstance(value, list):
                return tuple(value)
            return (value,)

        return value
