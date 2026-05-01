from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
import torchaudio

from src.utils.setuptools import RequiresSetupABCMeta, requires_setup

from .batch import Sample
from .preprocessing.audio import AudioPreprocessingUnit, MelSpectrogram
from .preprocessing.audio import Skip as AudioSkip
from .preprocessing.sequential import Pipeline, Skip
from .preprocessing.sequential.augmentations import Augmentation
from .preprocessing.sequential.transforms import Transform
from .raw_data import DataReader, DfData, Key, Record
from .schema import (
    CategoricalValueSpec,
    FieldSpec,
    OpaqueShapeSpec,
    OpaqueValueSpec,
    ReferenceValueSpec,
    ScalarShapeSpec,
    Schema,
    TensorShapeSpec,
    TensorValueSpec,
    TokenValueSpec,
)


class Dataset(torch_data.Dataset, ABC, metaclass=RequiresSetupABCMeta):
    def __init__(
        self,
        feature_data: DfData,
        schema: Schema,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
    ):
        self._samples_keys = samples_keys
        self._feature_data = feature_data
        self._schema = schema
        self._target_data = target_data

    def __len__(self):
        return self._len()

    def __getitem__(self, index: int) -> Sample:
        return self._getitem(index)

    def setup(self):
        if isinstance(self._feature_data, DataReader):
            self._feature_data.setup()

        if self._target_data is not None and isinstance(self._target_data, DataReader):
            self._target_data.setup()

        if self._samples_keys is None:
            self._samples_keys = {
                idx: {"key": key} for idx, key in enumerate(self._feature_data.get_keys())
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
    def _read_sample(self, index: int) -> Tuple[Key, Record, Optional[Record]]:
        raise NotImplementedError

    def _merge_records(self, feature_record: Record, target_record: Optional[Record]) -> Record:
        merged = dict(feature_record)
        if target_record is not None:
            merged.update(target_record)
        return merged

    def _pack_sample(self, record: Record) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fields: Dict[str, Any] = {}
        for field_name, field_spec in self._schema.fields.items():
            if field_name not in record:
                if field_spec.required:
                    raise KeyError(f"Missing required field '{field_name}' in sample record")
                continue
            fields[field_name] = self._pack_field_value(field_spec, record[field_name])

        meta = {key: value for key, value in record.items() if key not in self._schema.fields}
        return fields, meta

    @staticmethod
    def _pack_field_value(field_spec: FieldSpec, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(field_spec.value, (TensorValueSpec, CategoricalValueSpec, TokenValueSpec)):
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            tensor = tensor.to(dtype=field_spec.value.dtype)
            Dataset._validate_tensor_shape(field_spec, tensor)
            return tensor

        if isinstance(field_spec.value, (ReferenceValueSpec, OpaqueValueSpec)):
            return value

        raise TypeError(
            f"Unsupported value spec for field '{field_spec.name}': "
            f"{type(field_spec.value).__name__}"
        )

    @staticmethod
    def _validate_tensor_shape(field_spec: FieldSpec, tensor: torch.Tensor) -> None:
        if isinstance(field_spec.shape, ScalarShapeSpec):
            if tensor.ndim != 0:
                raise ValueError(f"Field '{field_spec.name}' expects a scalar tensor")
            return

        if isinstance(field_spec.shape, TensorShapeSpec):
            if tensor.ndim != len(field_spec.shape.axes):
                raise ValueError(
                    f"Field '{field_spec.name}' expects rank {len(field_spec.shape.axes)}, "
                    f"got rank {tensor.ndim}"
                )
            return

        if isinstance(field_spec.shape, OpaqueShapeSpec):
            return

        raise TypeError(
            f"Unsupported shape spec for field '{field_spec.name}': "
            f"{type(field_spec.shape).__name__}"
        )


class FlatDataset(Dataset):
    def _len(self):
        return len(self._samples_keys)

    def _getitem(self, index: int) -> Sample:
        sample_id, feature_record, target_record = self._read_sample(index)
        fields, meta = self._pack_sample(self._merge_records(feature_record, target_record))
        return Sample(sample_id=sample_id, fields=fields, meta=meta)

    def _read_sample(self, index: int) -> Tuple[Key, Record, Optional[Record]]:
        (key,) = self._samples_keys[index].values()
        feature_record = self._feature_data[key]
        target_record = self._target_data[key] if self._target_data is not None else None
        return key, feature_record, target_record


class SequentialDataset(Dataset):
    def __init__(
        self,
        feature_data: DfData,
        schema: Schema,
        sequential_features_key: str = "sequential_features",
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
        transforms: Optional[Pipeline[Transform]] = None,
        augmentations: Optional[Pipeline[Augmentation]] = None,
    ):
        super().__init__(feature_data, schema, samples_keys, target_data)
        self._sequential_features_key = sequential_features_key
        self._transforms = transforms if transforms is not None else Skip()
        self._augmentations = augmentations if augmentations is not None else Skip()

    def _len(self):
        return len(self._samples_keys)

    def _getitem(self, index: int) -> Sample:
        sample_id, feature_record, target_record = self._read_sample(index)
        sequential_features, meta = self._get_sequential_meta(feature_record)

        sequential_features = self._augmentations(sequential_features, meta)
        sequential_features = self._transforms(sequential_features, meta)

        merged_record = dict(meta)
        merged_record.update(sequential_features)
        if target_record is not None:
            merged_record.update(target_record)

        fields, sample_meta = self._pack_sample(merged_record)
        return Sample(sample_id=sample_id, fields=fields, meta=sample_meta)

    def _read_sample(self, index: int) -> Tuple[Key, Record, Optional[Record]]:
        (key,) = self._samples_keys[index].values()
        feature_record = self._feature_data[key]
        target_record = self._target_data[key] if self._target_data is not None else None
        return key, feature_record, target_record

    def _get_sequential_meta(self, feature_record: Record) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sequential_features = dict(feature_record[self._sequential_features_key])
        meta = {
            key: value
            for key, value in feature_record.items()
            if key != self._sequential_features_key
        }
        return sequential_features, meta


class BaseAudioDataset(Dataset):
    def __init__(
        self,
        feature_data: DfData,
        schema: Schema,
        audio_path_key: str = "audio_path",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = None,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
    ):
        super().__init__(feature_data, schema, samples_keys, target_data)
        self._audio_path_key = audio_path_key
        self._audio_root_dir = Path(audio_root_dir) if audio_root_dir is not None else None
        self._target_sr = target_sr
        self._clip_duration_seconds = clip_duration_seconds

    def _len(self):
        return len(self._samples_keys)

    def _read_sample(self, index: int) -> Tuple[Key, Record, Optional[Record]]:
        (key,) = self._samples_keys[index].values()
        feature_record = self._feature_data[key]
        target_record = self._target_data[key] if self._target_data is not None else None
        return key, feature_record, target_record

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        resolved_audio_path = self._resolve_audio_path(audio_path)
        waveform, sr = self._read_audio(resolved_audio_path)
        waveform = waveform[0:1, :]

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
        return (
            "libtorchcodec" in error_message
            or "torchcodec" in error_message
            or "ffmpeg" in error_message
        )

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
        candidate_paths = []

        if audio_path_obj.is_absolute():
            candidate_paths.append(audio_path_obj)
            normalized_legacy_path = self._normalize_legacy_absolute_path(audio_path_obj)
            if normalized_legacy_path is not None:
                candidate_paths.append(normalized_legacy_path)
        else:
            relative_audio_path = audio_path_obj
            if (
                self._audio_root_dir is not None
                and relative_audio_path.parts
                and relative_audio_path.parts[0] == self._audio_root_dir.name
            ):
                relative_audio_path = Path(*relative_audio_path.parts[1:])

            if self._audio_root_dir is not None:
                candidate_paths.append(self._audio_root_dir / relative_audio_path)
            candidate_paths.append(relative_audio_path)

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

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


class AudioDataset(BaseAudioDataset):
    def __init__(
        self,
        feature_data: DfData,
        schema: Schema,
        mel_spectrogram: Optional[MelSpectrogram],
        audio_path_key: str = "audio_path",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = None,
        waveform_field_name: Optional[str] = "waveform",
        spectrogram_field_name: Optional[str] = "mel_spectrogram",
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
        waveform_augmentations: Optional[AudioPreprocessingUnit] = None,
        spectrogram_augmentations: Optional[AudioPreprocessingUnit] = None,
    ):
        super().__init__(
            feature_data=feature_data,
            schema=schema,
            audio_path_key=audio_path_key,
            audio_root_dir=audio_root_dir,
            target_sr=target_sr,
            clip_duration_seconds=clip_duration_seconds,
            samples_keys=samples_keys,
            target_data=target_data,
        )
        self._mel_spectrogram = mel_spectrogram
        self._waveform_field_name = waveform_field_name
        self._spectrogram_field_name = spectrogram_field_name
        self._waveform_augmentations = (
            waveform_augmentations if waveform_augmentations is not None else AudioSkip()
        )
        self._spectrogram_augmentations = (
            spectrogram_augmentations if spectrogram_augmentations is not None else AudioSkip()
        )

    def _getitem(self, index: int) -> Sample:
        sample_id, feature_record, target_record = self._read_sample(index)

        waveform = self._load_audio(feature_record[self._audio_path_key])
        waveform = self._trim_or_pad_waveform(waveform)
        waveform = self._waveform_augmentations(waveform)

        merged_record = self._merge_records(feature_record, target_record)
        if self._waveform_field_name is not None:
            merged_record[self._waveform_field_name] = waveform.squeeze(0)

        if self._mel_spectrogram is not None and self._spectrogram_field_name is not None:
            spectrogram = self._mel_spectrogram(waveform)
            spectrogram = self._spectrogram_augmentations(spectrogram)
            merged_record[self._spectrogram_field_name] = spectrogram

        fields, meta = self._pack_sample(merged_record)
        return Sample(sample_id=sample_id, fields=fields, meta=meta)


class SourceSeparationDataset(BaseAudioDataset):
    def __init__(
        self,
        feature_data: DfData,
        schema: Schema,
        audio_path_key: str = "audio_path",
        source_audio_paths_key: str = "source_audio_paths",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = None,
        mixture_field_name: str = "mixture_audio",
        sources_field_name: str = "sources_audio",
        source_activity_field_name: Optional[str] = "source_activity",
        max_num_sources: int = 4,
        samples_keys: Optional[DfData] = None,
        target_data: Optional[DfData] = None,
    ):
        super().__init__(
            feature_data=feature_data,
            schema=schema,
            audio_path_key=audio_path_key,
            audio_root_dir=audio_root_dir,
            target_sr=target_sr,
            clip_duration_seconds=clip_duration_seconds,
            samples_keys=samples_keys,
            target_data=target_data,
        )
        self._source_audio_paths_key = source_audio_paths_key
        self._mixture_field_name = mixture_field_name
        self._sources_field_name = sources_field_name
        self._source_activity_field_name = source_activity_field_name
        self._max_num_sources = max_num_sources

    def _getitem(self, index: int) -> Sample:
        sample_id, feature_record, target_record = self._read_sample(index)
        if target_record is None:
            raise ValueError("SourceSeparationDataset requires target_data with source paths")

        mixture_waveform = self._load_audio(feature_record[self._audio_path_key])
        mixture_waveform = self._trim_or_pad_waveform(mixture_waveform)

        source_paths = tuple(target_record.get(self._source_audio_paths_key, ()))
        source_waveforms = self._load_sources(source_paths)
        source_activity = self._build_source_activity(len(source_paths))

        merged_record = self._merge_records(feature_record, target_record)
        merged_record[self._mixture_field_name] = mixture_waveform.squeeze(0)
        merged_record[self._sources_field_name] = source_waveforms
        if self._source_activity_field_name is not None:
            merged_record[self._source_activity_field_name] = source_activity

        fields, meta = self._pack_sample(merged_record)
        return Sample(sample_id=sample_id, fields=fields, meta=meta)

    def _load_sources(self, source_paths: Tuple[str, ...]) -> torch.Tensor:
        if len(source_paths) > self._max_num_sources:
            raise ValueError(
                f"Source sample contains {len(source_paths)} sources, "
                f"but max_num_sources={self._max_num_sources}"
            )

        source_waveforms: List[torch.Tensor] = []
        for source_path in source_paths:
            waveform = self._load_audio(source_path)
            waveform = self._trim_or_pad_waveform(waveform)
            source_waveforms.append(waveform.squeeze(0))

        if not source_waveforms:
            zero_waveform = torch.zeros(self._target_num_samples(), dtype=torch.float32)
            source_waveforms = [zero_waveform]

        padded_sources = list(source_waveforms)
        zero_source = torch.zeros_like(source_waveforms[0])
        while len(padded_sources) < self._max_num_sources:
            padded_sources.append(zero_source.clone())

        return torch.stack(padded_sources, dim=0)

    def _build_source_activity(self, num_sources: int) -> torch.Tensor:
        source_activity = torch.zeros(self._max_num_sources, dtype=torch.bool)
        if num_sources > 0:
            source_activity[:num_sources] = True
        return source_activity

    def _target_num_samples(self) -> int:
        if self._clip_duration_seconds is None:
            raise ValueError(
                "SourceSeparationDataset requires clip_duration_seconds "
                "when samples may contain no active sources"
            )
        return int(round(self._clip_duration_seconds * self._target_sr))
