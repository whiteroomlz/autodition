from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from .base import GenerateFeature, Map


class GenerateNumericalFeature(GenerateFeature, ABC):
    @abstractmethod
    def _reduce(self, bucket_to_reduce: np.array, meta: Dict) -> float:
        raise NotImplementedError


class Log10ClipShiftTransform(Map, GenerateNumericalFeature):
    shift: int

    def __init__(self, feature_to_apply: str, feature_to_generate: str, shift: int = 100):
        super().__init__(feature_to_apply, feature_to_generate)
        self.shift = shift

    def _map(self, features_to_map: np.array, meta: Dict) -> np.array:
        return np.log10(np.clip(features_to_map + self.shift, 1, None))


class Log10ClipShiftClipSignTransform(Map, GenerateNumericalFeature):
    def __init__(
        self,
        feature_to_apply: str,
        feature_to_generate: str,
        use_positive: bool = True,
        shift: int = 100,
    ):
        super().__init__(feature_to_apply, feature_to_generate)
        self.shift = shift
        self.use_positive = use_positive

    def _map(self, features_to_map: np.array, meta: Dict) -> np.array:
        signs = np.sign(features_to_map)
        log10 = np.log10(np.abs(features_to_map) + self.shift) * signs

        if self.use_positive:
            log10_clip = np.clip(log10, 0, None)
        else:
            log10_clip = np.clip(log10, None, 0)

        return log10_clip


class NumericalToCategoricalThroughQuantileTransform(Map, GenerateNumericalFeature):
    quantile: np.ndarray

    def __init__(self, feature_to_apply: str, feature_to_generate: str, quantile: np.ndarray):
        super().__init__(feature_to_apply, feature_to_generate)
        self.quantile = quantile

    def _map(self, features_to_map: np.array, meta: Dict) -> np.array:
        return np.searchsorted(self.quantile, features_to_map)
