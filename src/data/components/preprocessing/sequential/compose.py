from abc import ABC
from typing import Dict, Generic, Tuple, TypeVar

import numpy as np

from .augmentations import Augmentation
from .base import PreprocessingUnit
from .transforms import Transform

T = TypeVar("T", Transform, Augmentation, PreprocessingUnit)


class Compose(PreprocessingUnit, Generic[T], ABC):
    preprocessing_units: Tuple[T]

    def __init__(self, preprocessing_units: Tuple[T]) -> None:
        if len(preprocessing_units) <= 1:
            raise ValueError("At least two preprocessing units are required")

        self.preprocessing_units = preprocessing_units


class Pipeline(Compose[T]):
    def _apply(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> None:
        for preprocessing in self.preprocessing_units:
            preprocessing(sequential_features, meta)


class RandomChoice(Compose[T]):
    def __init__(self, preprocessing_units: Tuple[T]):
        super().__init__(preprocessing_units)
        self.candidates_count = len(self.preprocessing_units)

    def _apply(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> None:
        preprocessing_index = np.random.randint(0, self.candidates_count)
        self.preprocessing_units[preprocessing_index](sequential_features, meta)
