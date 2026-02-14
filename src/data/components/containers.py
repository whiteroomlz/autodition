from collections import namedtuple
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .raw_data import Key

Sample_ = namedtuple("Sample", ["id", "raw", "numerical", "categorical", "targets"])


class Sample(Sample_):
    def __new__(
        cls,
        sample_id: Key,
        raw: Optional[Any] = None,
        numerical: Optional[torch.FloatTensor] = None,
        categorical: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
    ):
        return super().__new__(
            cls,
            id=sample_id,  # noqa
            raw=raw,  # noqa
            numerical=numerical,  # noqa
            categorical=categorical,  # noqa
            targets=targets,  # noqa
        )


ModelBatch_ = namedtuple(
    "ModelBatch", ["sample_ids", "raw", "numerical", "categorical", "targets", "padding_mask"]
)


class ModelBatch(ModelBatch_):
    def __new__(
        cls,
        sample_ids: Tuple[Key],
        raw: Optional[Tuple[Any]] = None,
        numerical: Optional[torch.FloatTensor] = None,
        categorical: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ):
        return super().__new__(
            cls,
            sample_ids=sample_ids,  # noqa
            raw=raw,  # noqa
            numerical=numerical,  # noqa
            categorical=categorical,  # noqa
            targets=targets,  # noqa
            padding_mask=padding_mask,  # noqa
        )


@dataclass
class FeatureTypeInfo:
    feature_names: List[str]


@dataclass
class TorchFeatureTypeInfo(FeatureTypeInfo):
    torch_dtype: torch.dtype


@dataclass
class NumericalFeatureInfo(TorchFeatureTypeInfo):
    ...


@dataclass
class CategoricalFeatureInfo(TorchFeatureTypeInfo):
    vocabularies_size: List[int]
    embeddings_dim: List[int]


@dataclass
class FeatureSchema:
    raw: Optional[FeatureTypeInfo] = None
    numerical: Optional[NumericalFeatureInfo] = None
    categorical: Optional[CategoricalFeatureInfo] = None

    feature_types: List[str] = field(default_factory=list)
    possible_feature_types: List[str] = field(default_factory=list)
    feature_info_dict: Dict[str, FeatureTypeInfo] = field(default_factory=dict)

    def __post_init__(self):
        self.feature_types = list()
        self.possible_feature_types = list()
        self.feature_info_dict = dict()

        for field_ in fields(self):
            if field_.default is None:
                field_value = getattr(self, field_.name)
                if field_value is not None:
                    self.feature_types.append(field_.name)
                    self.feature_info_dict[field_.name] = field_value
                self.possible_feature_types.append(field_.name)

    def __getitem__(self, feature_type: str) -> FeatureTypeInfo:
        return self.feature_info_dict[feature_type]


@dataclass
class TargetSchema(FeatureSchema):
    ...


@dataclass
class SequentialFeatureSchema(FeatureSchema):
    sequential_features_key: str = field(default_factory=str)
