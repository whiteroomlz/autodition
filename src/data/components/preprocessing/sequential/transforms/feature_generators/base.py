from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict, Tuple, TypeAlias, Union

import numpy as np

from ..base import Transform

ValueType: TypeAlias = Union[int, float]


class GenerateFeature(Transform, ABC):
    features_to_apply: Union[str, Tuple[str]]
    feature_to_generate: str

    def __init__(self, features_to_apply: Union[str, Tuple[str]], feature_to_generate: str):
        if isinstance(features_to_apply, Tuple):
            self.features_to_apply = features_to_apply
        else:
            self.features_to_apply = (features_to_apply,)
        self.feature_to_generate = feature_to_generate

    def __call__(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> Dict:
        self._apply(sequential_features, meta)
        return sequential_features

    def _apply(self, sequential_features: Dict[str, np.ndarray], meta) -> None:
        features_to_apply = np.vstack(itemgetter(*self.features_to_apply)(sequential_features))
        mapped = self._map(features_to_apply, meta)
        if len(mapped.shape) > 1:
            reduced = np.apply_along_axis(self._reduce, axis=1, arr=mapped, meta=meta)
            sequential_features[self.feature_to_generate] = reduced
        else:
            sequential_features[self.feature_to_generate] = mapped

    @abstractmethod
    def _map(self, features_to_map: np.array, meta: Dict) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def _reduce(self, bucket_to_reduce: np.array, meta: Dict) -> ValueType:
        raise NotImplementedError


class Map(GenerateFeature, ABC):
    def __init__(self, feature_to_apply: str, feature_to_generate: str):
        super().__init__(feature_to_apply, feature_to_generate)

    def _reduce(self, bucket_to_reduce: np.array, meta: Dict) -> ValueType:
        raise NotImplementedError


class Reduce(GenerateFeature, ABC):
    def _map(self, features_to_map: np.array, meta: Dict) -> np.array:
        raise features_to_map
