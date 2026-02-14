from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from .base import ApplyMask
from .utils import get_sequence_length


class Crop(ApplyMask, ABC):
    invert: bool

    def __init__(self, invert: bool = False):
        self.invert = invert

    def _get_mask(self, sequential_features: Dict, meta: Dict) -> np.ndarray:
        sequence_length = get_sequence_length(sequential_features)

        left_bound_index, right_bound_index = self._get_bounds(sequence_length, meta)
        mask = np.zeros(sequence_length).astype(bool)

        if not self.invert:
            mask[:left_bound_index] = True
            mask[right_bound_index:] = True
        else:
            mask[left_bound_index:right_bound_index] = True

        return mask

    @abstractmethod
    def _get_bounds(self, sequence_length: int, meta: Dict) -> Tuple[int, int]:
        raise NotImplementedError


class StaticCrop(Crop):
    left_bound_index: int
    right_bound_index: int

    def __init__(self, left_bound_index: int, right_bound_index: int, invert: bool = False):
        super().__init__(invert)

        if not (0 <= left_bound_index < right_bound_index):
            raise ValueError(f"Incorrect bounds passed - {left_bound_index}:{right_bound_index}")
        self.left_bound_index = left_bound_index
        self.right_bound_index = right_bound_index

    def _get_bounds(self, sequence_length: int, meta: Dict) -> Tuple[int, int]:
        return self.left_bound_index, self.right_bound_index


class ScaleCrop(Crop):
    left_bound_scale: float
    right_bound_scale: float

    def __init__(
        self, left_bound_scale: float, right_bound_scale: float = 1, invert: bool = False
    ):
        super().__init__(invert)

        if not (0 <= left_bound_scale < right_bound_scale <= 1):
            raise ValueError(f"Incorrect bounds passed - {left_bound_scale}:{right_bound_scale}")
        self.left_bound_scale = left_bound_scale
        self.right_bound_scale = right_bound_scale

    def _get_bounds(self, sequence_length: int, meta: Dict) -> Tuple[int, int]:
        left_bound_index = int(sequence_length * self.left_bound_scale)
        right_bound_index = int(sequence_length * self.right_bound_scale)
        return left_bound_index, right_bound_index


class UniformScaleRandomCrop(ScaleCrop):
    min_left_bound_scale: float
    max_left_bound_scale: float

    def __init__(
        self,
        max_diff: float,
        left_bound_scale: float,
        right_bound_scale: float = 1,
        invert: bool = False,
    ):
        super().__init__(left_bound_scale, right_bound_scale, invert)

        if not (
            0
            <= left_bound_scale - max_diff
            < left_bound_scale + max_diff
            < self.right_bound_scale
            <= 1
        ):
            raise ValueError(f"Incorrect max_diff passed - {max_diff}")
        self.min_left_bound_scale = left_bound_scale - max_diff
        self.max_left_bound_scale = left_bound_scale + max_diff

    def _get_bounds(self, sequence_length: int, meta: Dict) -> Tuple[int, int]:
        left_bound_index = int(
            np.random.uniform(self.min_left_bound_scale, self.max_left_bound_scale)
            * sequence_length
        )
        right_bound_index = int(sequence_length * self.right_bound_scale)
        return left_bound_index, right_bound_index


class NormalScaleRandomCrop(ScaleCrop):
    max_diff: float

    def __init__(
        self,
        max_diff: float,
        left_bound_scale: float,
        right_bound_scale: float = 1,
        invert: bool = False,
    ):
        super().__init__(left_bound_scale, right_bound_scale, invert)

        if not (
            0
            <= left_bound_scale - max_diff
            < left_bound_scale + max_diff
            < self.right_bound_scale
            <= 1
        ):
            raise ValueError(f"Incorrect max_diff passed - {max_diff}")
        self.max_diff = max_diff

    def _get_bounds(self, sequence_length: int, meta: Dict) -> Tuple[int, int]:
        left_bound_index = int(
            np.random.normal(loc=self.left_bound_scale, scale=(self.max_diff / 5))
            * sequence_length
        )
        right_bound_index = int(sequence_length * self.right_bound_scale)
        return left_bound_index, right_bound_index


class MetaInfoCrop(Crop):
    meta_left_bound_key: str
    meta_right_bound_key: str

    def __init__(self, meta_left_bound_key: str, meta_right_bound_key: str, invert: bool = False):
        super().__init__(invert)
        self.meta_left_bound_key = meta_left_bound_key
        self.meta_right_bound_key = meta_right_bound_key

    def _get_bounds(self, sequence_length: int, meta: Dict) -> Tuple[int, int]:
        left_bound_index = meta[self.meta_left_bound_key]
        right_bound_index = meta[self.meta_right_bound_key]
        return left_bound_index, right_bound_index
