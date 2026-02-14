from abc import ABC
from typing import Dict

import numpy as np

from ..base import Augmentation
from .utils import get_sequence_length


class ApplyMask(Augmentation, ABC):
    def _apply(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> None:
        mask = self._get_mask(sequential_features, meta)
        for key in sequential_features.keys():
            sequential_features[key] = sequential_features[key][mask]

    def _get_mask(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> np.ndarray:
        raise NotImplementedError


class Sort(ApplyMask):
    def __init__(self, sort_key: str, ascending: bool = True):
        self.sort_key = sort_key
        self.ascending = ascending

    def _get_mask(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> np.ndarray:
        mask = np.argsort(sequential_features[self.sort_key])
        if not self.ascending:
            mask = mask[::-1]

        return mask


class RandomDuplicate(ApplyMask):
    duplicate_probability: float

    def __init__(self, duplicate_probability: float):
        self.duplicate_probability = duplicate_probability

    def _get_mask(self, sequential_features: Dict[str, np.ndarray], meta: Dict) -> np.ndarray:
        sequence_length = get_sequence_length(sequential_features)

        duplication_mask = (
            np.random.binomial(n=1, p=self.duplicate_probability, size=sequence_length) + 1
        )
        mask_to_apply = np.repeat(np.arange(sequence_length), duplication_mask, axis=0)

        return mask_to_apply
