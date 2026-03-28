from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
import torchaudio

from src.utils.setuptools import RequiresSetupABCMeta, requires_setup

from .containers import (
    FeatureSchema,
    Sample,
    SequentialFeatureSchema,
    TargetSchema,
    TorchFeatureTypeInfo,
)
from .preprocessing.audio import AudioPreprocessingUnit, MelSpectrogram
from .preprocessing.audio import Skip as AudioSkip
from .preprocessing.sequential import Pipeline, Skip
from .preprocessing.sequential.augmentations import Augmentation
from .preprocessing.sequential.transforms import Transform
from .raw_data import DataReader, DfData, Key, Record

# region abstract.


class Dataset(torch_data.Dataset, ABC, metaclass=RequiresSetupABCMeta):
    def __init__(
        self,
        feature_data: DfData,
        feature_schema: FeatureSchema,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
        target_schema: Optional[TargetSchema] = None,
    ):
        self._samples_keys = samples_keys

        self._feature_data = feature_data
        self._feature_schema = feature_schema

        self._target_data = target_data
        self._target_schema = target_schema

    def __len__(self):
        return self._len()

    def __getitem__(self, index: int) -> Sample:
        return self._getitem(index)

    def setup(self):
        if isinstance(self._feature_data, DataReader):
            self._feature_data.setup()

        if self._target_data is not None:
            if isinstance(self._target_data, DataReader):
                self._target_data.setup()

        if self._samples_keys is None:
            self._samples_keys = {
                idx: dict(key=key) for idx, key in enumerate(self._feature_data.get_keys())
            }
        elif isinstance(self._samples_keys, DataReader):
            self._samples_keys.setup()

    @requires_setup
    @abstractmethod
    def _len(self):
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def _getitem(self, index: int) -> Sample:
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def _read_sample(self, index: int) -> Tuple[Key, Record, Record]:
        raise NotImplementedError


# endregion


class FlatDataset(Dataset):
    def __init__(
        self,
        feature_data: DfData,
        feature_schema: FeatureSchema,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
        target_schema: Optional[TargetSchema] = None,
    ):
        super().__init__(feature_data, feature_schema, samples_keys, target_data, target_schema)

        if target_schema:
            self._target_filter = set()
            for target_type in target_schema.feature_types:
                self._target_filter.update(target_schema[target_type].feature_names)

    def _len(self):
        return len(self._samples_keys)

    def _getitem(self, index: int) -> Sample:
        sample_id, features, targets = self._read_sample(index)

        if targets is not None:
            self._filter_targets(targets)

        features = pack_flat_features(self._feature_schema, features)
        if targets is not None:
            targets = pack_flat_targets(self._target_schema, targets)

        sample = Sample(sample_id=sample_id, raw=features["raw"], targets=targets)

        return sample

    def _read_sample(self, index: int) -> Tuple[Key, Record, Record]:
        (key,) = self._samples_keys[index].values()
        features = self._feature_data[key]

        if self._target_data is not None:
            targets = self._target_data[key]
        else:
            targets = None

        return key, features, targets

    def _filter_targets(self, targets: Record) -> None:
        keys = tuple(targets.keys())
        for key in keys:
            if key not in self._target_filter:
                del targets[key]


class SequentialDataset(Dataset):
    def __init__(
        self,
        feature_data: DfData,
        feature_schema: SequentialFeatureSchema,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
        target_schema: Optional[TargetSchema] = None,
        transforms: Optional[Pipeline[Transform]] = None,
        augmentations: Optional[Pipeline[Augmentation]] = None,
    ):
        super().__init__(feature_data, feature_schema, samples_keys, target_data, target_schema)

        self._feature_filter = set()
        for feature_type in feature_schema.feature_types:
            self._feature_filter.update(feature_schema[feature_type].feature_names)
        self._sequential_features_key = feature_schema.sequential_features_key

        if target_schema:
            self._target_filter = set()
            for target_type in target_schema.feature_types:
                self._target_filter.update(target_schema[target_type].feature_names)

        if transforms is None:
            self._transforms = Skip()
        else:
            self._transforms = transforms

        if augmentations is None:
            self._augmentations = Skip()
        else:
            self._augmentations = augmentations

    def _len(self):
        return len(self._samples_keys)

    def _getitem(self, index: int) -> Sample:
        sample_id, features, targets = self._read_sample(index)

        sequential_features, meta = self._get_sequential_meta(features)

        # The order is crucially important
        sequential_features = self._augmentations(sequential_features, meta)
        sequential_features = self._transforms(sequential_features, meta)
        self._filter_features(sequential_features)

        if targets is not None:
            self._filter_targets(targets)

        features = pack_sequential_features(self._feature_schema, sequential_features)
        if targets is not None:
            targets = pack_flat_targets(self._target_schema, targets)

        sample = Sample(
            sample_id=sample_id,
            numerical=features["numerical"],
            categorical=features["categorical"],
            targets=targets,
        )

        return sample

    def _read_sample(self, index: int) -> Tuple[Key, Record, Record]:
        (key,) = self._samples_keys[index].values()
        features = self._feature_data[key]

        if self._target_data is not None:
            targets = self._target_data[key]
        else:
            targets = None

        return key, features, targets

    def _get_sequential_meta(
        self, features: Record
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        sequential_features = features[self._sequential_features_key]
        meta = {
            key: value for key, value in features.items() if key != self._sequential_features_key
        }
        return sequential_features, meta

    def _filter_features(self, sequential_features: Dict[str, np.ndarray]) -> None:
        keys = tuple(sequential_features.keys())
        for key in keys:
            if key not in self._feature_filter:
                sequential_features.pop(key)

    def _filter_targets(self, targets: Record) -> None:
        keys = tuple(targets.keys())
        for key in keys:
            if key not in self._target_filter:
                del targets[key]


class AudioDataset(Dataset):
    def __init__(
        self,
        feature_data: DfData,
        feature_schema: FeatureSchema,
        mel_spectrogram: Optional[MelSpectrogram],
        audio_path_key: str = "audio_path",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = None,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
        target_schema: Optional[TargetSchema] = None,
        waveform_augmentations: Optional[AudioPreprocessingUnit] = None,
        spectrogram_augmentations: Optional[AudioPreprocessingUnit] = None,
        return_waveform_in_sample: bool = False,
    ):
        super().__init__(feature_data, feature_schema, samples_keys, target_data, target_schema)

        self._audio_path_key = audio_path_key
        self._audio_root_dir = Path(audio_root_dir) if audio_root_dir is not None else None
        self._target_sr = target_sr
        self._mel_spectrogram = mel_spectrogram
        self._clip_duration_seconds = clip_duration_seconds
        self._return_waveform_in_sample = return_waveform_in_sample

        self._waveform_augmentations = waveform_augmentations if waveform_augmentations is not None else AudioSkip()
        self._spectrogram_augmentations = spectrogram_augmentations if spectrogram_augmentations is not None else AudioSkip()

        if target_schema:
            self._target_filter = set()
            for target_type in target_schema.feature_types:
                self._target_filter.update(target_schema[target_type].feature_names)

    def _len(self):
        return len(self._samples_keys)

    def _getitem(self, index: int) -> Sample:
        sample_id, features, targets = self._read_sample(index)

        waveform = self._load_audio(features[self._audio_path_key])
        waveform = self._trim_or_pad_waveform(waveform)
        waveform = self._waveform_augmentations(waveform)

        if self._mel_spectrogram is not None:
            spectrogram = self._mel_spectrogram(waveform)
            spectrogram = self._spectrogram_augmentations(spectrogram)
        else:
            spectrogram = None

        if targets is not None:
            self._filter_targets(targets)
            targets = pack_flat_targets(self._target_schema, targets)

        raw_waveform = waveform.squeeze(0) if self._return_waveform_in_sample else None
        sample = Sample(
            sample_id=sample_id,
            raw=raw_waveform,
            numerical=spectrogram,
            targets=targets,
        )

        return sample

    def _read_sample(self, index: int) -> Tuple[Key, Record, Record]:
        (key,) = self._samples_keys[index].values()
        features = self._feature_data[key]

        if self._target_data is not None:
            targets = self._target_data[key]
        else:
            targets = None

        return key, features, targets

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        resolved_audio_path = self._resolve_audio_path(audio_path)
        waveform, sr = self._read_audio(resolved_audio_path)
        waveform = waveform[0:1, :]  # mono

        if sr != self._target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self._target_sr)

        return waveform

    def _read_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        try:
            return torchaudio.load(str(audio_path))
        except RuntimeError as error:
            if not self._should_use_soundfile_fallback(error):
                raise

        waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return torch.from_numpy(waveform.T), sr

    @staticmethod
    def _should_use_soundfile_fallback(error: RuntimeError) -> bool:
        error_message = str(error).lower()
        return "libtorchcodec" in error_message or "torchcodec" in error_message or "ffmpeg" in error_message

    def _trim_or_pad_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._clip_duration_seconds is None:
            return waveform

        max_num_samples = int(round(self._clip_duration_seconds * self._target_sr))
        current_num_samples = waveform.shape[-1]

        if current_num_samples > max_num_samples:
            return waveform[..., :max_num_samples]

        if current_num_samples < max_num_samples:
            return F.pad(waveform, (0, max_num_samples - current_num_samples))

        return waveform

    def _resolve_audio_path(self, audio_path: str) -> Path:
        audio_path_obj = Path(audio_path)

        if audio_path_obj.is_absolute():
            if audio_path_obj.exists():
                return audio_path_obj

            normalized_legacy_path = self._normalize_legacy_absolute_path(audio_path_obj)
            if normalized_legacy_path is not None and normalized_legacy_path.exists():
                return normalized_legacy_path
        else:
            relative_audio_path = audio_path_obj
            if (
                self._audio_root_dir is not None
                and relative_audio_path.parts
                and relative_audio_path.parts[0] == self._audio_root_dir.name
            ):
                relative_audio_path = Path(*relative_audio_path.parts[1:])

            if self._audio_root_dir is not None:
                resolved_audio_path = self._audio_root_dir / relative_audio_path
            else:
                resolved_audio_path = relative_audio_path

            if resolved_audio_path.exists():
                return resolved_audio_path

        raise FileNotFoundError(
            f"Unable to resolve audio path '{audio_path}'. "
            f"Configured audio_root_dir='{self._audio_root_dir}'."
        )

    def _normalize_legacy_absolute_path(self, audio_path: Path) -> Optional[Path]:
        if self._audio_root_dir is None:
            return None

        try:
            dataset_root_index = audio_path.parts.index(self._audio_root_dir.name)
        except ValueError:
            return None

        relative_audio_path = Path(*audio_path.parts[dataset_root_index + 1 :])
        return self._audio_root_dir / relative_audio_path

    def _filter_targets(self, targets: Record) -> None:
        keys = tuple(targets.keys())
        for key in keys:
            if key not in self._target_filter:
                del targets[key]


def pack_flat_features(feature_schema: FeatureSchema, features: Dict):
    features_ = dict()

    for feature_type in feature_schema.possible_feature_types:
        if feature_type in feature_schema.feature_types:
            features_[feature_type] = [
                features[feature_name]
                for feature_name in feature_schema[feature_type].feature_names
            ]

            feature_info = feature_schema[feature_type]
            if isinstance(feature_info, TorchFeatureTypeInfo):
                features_[feature_type] = torch.tensor(features_[feature_type]).to(
                    feature_info.torch_dtype
                )
        else:
            features_[feature_type] = None

    return features_


def pack_flat_targets(target_schema: TargetSchema, targets: Dict):
    targets_ = dict()

    for feature_type in target_schema.feature_types:
        for feature_name in target_schema[feature_type].feature_names:

            feature_info = target_schema[feature_type]

            if isinstance(feature_info, TorchFeatureTypeInfo):
                targets_[feature_name] = torch.tensor(targets[feature_name]).to(
                    feature_info.torch_dtype
                )
            else:
                targets_[feature_name] = targets[feature_name]

    return targets_


def pack_sequential_features(feature_schema: FeatureSchema, features: Dict):
    features_ = dict()

    for feature_type in feature_schema.possible_feature_types:
        if feature_type in feature_schema.feature_types:
            features_[feature_type] = list()
            feature_info = feature_schema[feature_type]
            torch_flag = isinstance(feature_info, TorchFeatureTypeInfo)

            for feature_name in feature_schema[feature_type].feature_names:
                feature = features[feature_name]

                if torch_flag:
                    feature = torch.tensor(feature).to(feature_info.torch_dtype)  # noqa
                    if feature.dim() == 1:
                        feature = feature[:, None]

                features_[feature_type].append(feature)

            if torch_flag:
                features_[feature_type] = torch.cat((features_[feature_type]), dim=1)
        else:
            features_[feature_type] = None

    return features_
