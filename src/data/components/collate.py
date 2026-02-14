from abc import ABC, abstractmethod
from typing import List

import torch

from .containers import FeatureSchema, ModelBatch, Sample, TargetSchema


class Collator(ABC):
    def __init__(self, feature_schema: FeatureSchema, target_schema: TargetSchema):
        self.feature_schema = feature_schema
        self.target_schema = target_schema

    def __call__(self, samples: List[Sample]) -> ModelBatch:
        raise NotImplementedError


class FlatCollator(Collator):
    def __call__(self, samples: List[Sample]) -> ModelBatch:
        sample_ids, raw, numerical, categorical, *targets_values = zip(
            *(
                (
                    sample.id,
                    sample.raw,
                    sample.numerical,
                    sample.categorical,
                    *(sample.targets.values() if sample.targets else (None,)),
                )
                for sample in samples
            )
        )

        if any(x is not None for x in numerical):
            numerical = torch.vstack(numerical).to(self.feature_schema.numerical.torch_dtype)
        else:
            numerical = None

        if any(x is not None for x in categorical):
            categorical = torch.vstack(categorical).to(self.feature_schema.categorical.torch_dtype)
        else:
            categorical = None

        if self.target_schema is not None:
            targets = dict()
            for target_key, target_values in zip(samples[0].targets.keys(), targets_values):
                if not all(value is None for value in target_values):
                    if (
                        self.target_schema.numerical
                        and target_key in self.target_schema.numerical.feature_names
                    ):
                        targets[target_key] = torch.vstack(target_values).to(
                            self.target_schema.numerical.torch_dtype
                        )
                    elif (
                        self.target_schema.categorical
                        and target_key in self.target_schema.categorical.feature_names
                    ):
                        targets[target_key] = torch.vstack(target_values).to(
                            self.target_schema.categorical.torch_dtype
                        )
                    else:
                        targets[target_key] = target_values
                else:
                    targets[target_key] = None
        else:
            targets = None

        return ModelBatch(
            sample_ids=sample_ids,
            raw=raw,
            numerical=numerical,
            categorical=categorical,
            targets=targets,
        )


class SequentialCollator(Collator):
    def __call__(self, samples: List[Sample]) -> ModelBatch:
        sample_ids, raw, numerical, categorical, sequence_lengths, *targets_values = zip(
            *(
                (
                    sample.id,
                    sample.raw,
                    sample.numerical,
                    sample.categorical,
                    (
                        len(sample.numerical)
                        if sample.numerical is not None
                        else (
                            len(sample.categorical)
                            if sample.categorical is not None
                            else len(sample.raw)
                        )
                    ),
                    *(sample.targets.values() if sample.targets else (None,)),
                )
                for sample in samples
            )
        )

        if any(x is not None for x in numerical):
            numerical = self.pad_tensors(numerical).to(self.feature_schema.numerical.torch_dtype)
        else:
            numerical = None

        if any(x is not None for x in categorical):
            categorical = self.pad_tensors(categorical).to(
                self.feature_schema.categorical.torch_dtype
            )
        else:
            categorical = None

        # TODO: think about duplicated code.
        if self.target_schema is not None:
            targets = dict()
            for target_key, target_values in zip(samples[0].targets.keys(), targets_values):
                if not all(value is None for value in target_values):
                    if (
                        self.target_schema.numerical
                        and target_key in self.target_schema.numerical.feature_names
                    ):
                        targets[target_key] = torch.vstack(target_values).to(
                            self.target_schema.numerical.torch_dtype
                        )
                    elif (
                        self.target_schema.categorical
                        and target_key in self.target_schema.categorical.feature_names
                    ):
                        targets[target_key] = torch.vstack(target_values).to(
                            self.target_schema.categorical.torch_dtype
                        )
                    else:
                        targets[target_key] = target_values
                else:
                    targets[target_key] = None

        padding_mask = self.create_padding_mask(sequence_lengths)

        return ModelBatch(
            sample_ids=sample_ids,
            numerical=numerical,
            categorical=categorical,
            targets=targets,
            padding_mask=padding_mask,
        )

    @abstractmethod
    def pad_tensors(self, tensors_to_pad) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def create_padding_mask(self, sequence_lengths: List[int]) -> torch.BoolTensor:
        raise NotImplementedError


class StaticLengthCollator(SequentialCollator):
    def __init__(
        self, feature_schema: FeatureSchema, target_schema: TargetSchema, max_sequence_length: int
    ) -> None:
        super().__init__(feature_schema, target_schema)
        self.max_sequence_length = max_sequence_length

    def pad_tensors(self, tensors_to_pad) -> torch.Tensor:
        padded_data = torch.nn.utils.rnn.pad_sequence(tensors_to_pad, batch_first=True)
        padded_data = torch.nn.functional.pad(
            padded_data, (0, 0, 0, self.max_sequence_length - padded_data.shape[1])
        )
        return padded_data

    def create_padding_mask(self, sequence_lengths: List[int]) -> torch.BoolTensor:
        max_length = self.max_sequence_length
        padding_mask = (
            torch.arange(max_length)[None, :] < torch.tensor(sequence_lengths)[:, None]
        ).bool()
        return padding_mask  # noqa


class DynamicLengthCollator(SequentialCollator):
    def pad_tensors(self, tensors_to_pad) -> torch.Tensor:
        padded_data = torch.nn.utils.rnn.pad_sequence(tensors_to_pad, batch_first=True)
        return padded_data

    def create_padding_mask(self, sequence_lengths: List[int]) -> torch.BoolTensor:
        max_length = max(sequence_lengths)
        padding_mask = (
            torch.arange(max_length)[None, :] < torch.tensor(sequence_lengths)[:, None]
        ).bool()
        return padding_mask  # noqa
