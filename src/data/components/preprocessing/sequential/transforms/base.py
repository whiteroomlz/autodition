from abc import ABC
from typing import Dict

import numpy as np

from ..base import PreprocessingUnit


class Transform(PreprocessingUnit, ABC):
    ...


class DeleteFeature(PreprocessingUnit):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def _apply(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> None:
        del sequential_features[self.feature_name]
