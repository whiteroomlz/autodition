from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch

from .raw_data import Key


@dataclass
class Sample:
    sample_id: Key
    fields: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, field_name: str) -> Any:
        return self.fields[field_name]

    def get(self, field_name: str, default: Any = None) -> Any:
        return self.fields.get(field_name, default)


@dataclass
class Batch:
    sample_ids: Tuple[Key, ...]
    fields: Dict[str, Any]
    masks: Dict[str, torch.BoolTensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, field_name: str) -> Any:
        return self.fields[field_name]

    def get(self, field_name: str, default: Any = None) -> Any:
        return self.fields.get(field_name, default)
