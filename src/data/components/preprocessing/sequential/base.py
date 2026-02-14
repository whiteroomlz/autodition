from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class PreprocessingUnit(ABC):
    def __call__(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> Dict:
        self._apply(sequential_features, meta)
        return sequential_features

    @abstractmethod
    def _apply(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> None:
        raise NotImplementedError


class Skip(PreprocessingUnit):
    def _apply(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> None:
        pass
