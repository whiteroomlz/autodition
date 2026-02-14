from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

from .base import ApplyMask
from .utils import get_sequence_length


class Dropout(ApplyMask, ABC):
    def _get_mask(self, sequential_features: Dict, meta: Dict) -> np.ndarray:
        sequence_length = get_sequence_length(sequential_features)
        prob_weights = self._get_probabilities(sequence_length, meta)

        mask = np.random.binomial(n=1, p=(1 - prob_weights), size=sequence_length).astype(bool)

        return mask

    @abstractmethod
    def _get_probabilities(self, sequence_length: int, meta: Dict) -> Union[float, np.ndarray]:
        raise NotImplementedError


class StaticDropout(Dropout):
    dropout_probability: float

    def __init__(self, dropout_probability: float):
        if not (0 <= dropout_probability <= 1):
            raise ValueError(f"Incorrect dropout_probability passed - {dropout_probability}")
        self.dropout_probability = dropout_probability

    def _get_probabilities(self, sequence_length: int, meta: Dict) -> Union[float, np.ndarray]:
        probabilities = self.dropout_probability
        return probabilities


class UniformDropout(Dropout):
    min_dropout_probability: float
    max_dropout_probability: float

    def __init__(self, min_dropout_probability: float, max_dropout_probability: float):
        if not (0 <= min_dropout_probability <= max_dropout_probability <= 1):
            raise ValueError(
                "Incorrect dropout_probabilities passed - {}:{}".format(
                    min_dropout_probability, max_dropout_probability
                )
            )
        self.min_dropout_probability = min_dropout_probability
        self.max_dropout_probability = max_dropout_probability

    def _get_probabilities(self, sequence_length: int, meta: Dict) -> Union[float, np.ndarray]:
        probabilities = np.random.uniform(
            self.min_dropout_probability, self.max_dropout_probability
        )
        return probabilities


class LinearWeightedDropout(Dropout):
    sampler_bias: float

    def __init__(
        self, dropout_probability: float, sampler_bias: float, use_cos_bias_scaling: bool = True
    ):
        super().__init__(dropout_probability)

        if not (0 <= sampler_bias <= 1):
            raise ValueError(f"Incorrect sampler_bias passed - {sampler_bias}")
        self.sampler_bias = sampler_bias

        if use_cos_bias_scaling:
            self.sampler_bias = (np.cos(np.pi * (self.sampler_bias + 1)) + 1) / 2

    def _get_probabilities(self, sequence_length: int, meta: Dict) -> Union[float, np.ndarray]:
        indices = np.arange(sequence_length)
        max_index = sequence_length - 1
        unscaled_probabilities = 1 / (
            self.sampler_bias * max_index + (1 - 2 * self.sampler_bias) * indices + 1
        )
        probabilities = unscaled_probabilities / unscaled_probabilities.sum()
        return probabilities


class BoundedDropout(Dropout):
    wrapped_dropout: Dropout
    left_bound: float
    right_bound: float
    invert: bool

    def __init__(
        self, dropout: Dropout, left_bound: float = 0, right_bound: float = 1, invert: bool = False
    ):
        self.wrapped_dropout = dropout

        if not (0 <= left_bound < right_bound <= 1):
            raise ValueError(f"Incorrect bounds passed - {left_bound}, {right_bound}")
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.invert = invert

    def _get_mask(self, sequential_features: Dict, meta: Dict) -> np.ndarray:
        mask = super()._get_mask(sequential_features, meta)

        sequence_length = get_sequence_length(sequential_features)
        left_bound_index = int(sequence_length * self.left_bound)
        right_bound_index = int(sequence_length * self.right_bound)

        if not self.invert:
            mask[:left_bound_index] = True
            mask[right_bound_index:] = True
        else:
            mask[left_bound_index:right_bound_index] = True

        return mask

    def _get_probabilities(self, sequence_length: int, meta: Dict) -> Union[float, np.ndarray]:
        return self.wrapped_dropout._get_probabilities(sequence_length, meta)
