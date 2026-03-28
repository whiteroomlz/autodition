import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.components.batch import Sample
from src.data.components.dataset import AudioDataset, SourceSeparationDataset
from src.data.components.raw_data import DfData
from src.data.components.schema import (
    CategoricalValueSpec,
    FieldSpec,
    ListBatchingSpec,
    ScalarShapeSpec,
    Schema,
    StackBatchingSpec,
    TensorShapeSpec,
    TensorValueSpec,
)


class MemoryData(DfData):
    def __init__(self, data):
        self.data = data

    def _getitem(self, key):
        return self.data[key]

    def _contains(self, key):
        return key in self.data

    def _len(self):
        return len(self.data)

    def _hash(self):
        return hash(tuple(sorted(self.data.keys())))

    def get_keys(self):
        return set(self.data.keys())


def write_wav(audio_path: Path, sample_rate: int = 16000, num_samples: int = 16000) -> None:
    waveform = (0.1 * np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate) * 32767).astype(
        np.int16
    )

    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(waveform.tobytes())


def build_dataset(audio_root_dir: Path, audio_path: str) -> AudioDataset:
    feature_data = MemoryData({"sample": {"audio_path": audio_path}})
    target_data = MemoryData({"sample": {"class_id": 3}})
    samples_keys = MemoryData({0: {"key": "sample"}})

    schema = Schema(
        fields={
            "waveform": FieldSpec(
                name="waveform",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=ListBatchingSpec(),
            ),
            "class_id": FieldSpec(
                name="class_id",
                role="supervision",
                value=CategoricalValueSpec(dtype=torch.long, cardinality=10),
                shape=ScalarShapeSpec(),
                batching=StackBatchingSpec(),
            ),
        }
    )

    dataset = AudioDataset(
        feature_data=feature_data,
        schema=schema,
        mel_spectrogram=None,
        audio_root_dir=str(audio_root_dir),
        target_sr=16000,
        samples_keys=samples_keys,
        target_data=target_data,
    )
    dataset.setup()
    return dataset


def build_separation_dataset(audio_root_dir: Path, source_audio_paths: tuple[str, ...]) -> SourceSeparationDataset:
    feature_data = MemoryData({"sample": {"audio_path": "audio/fold1/example.wav"}})
    target_data = MemoryData({"sample": {"source_audio_paths": source_audio_paths}})
    samples_keys = MemoryData({0: {"key": "sample"}})

    schema = Schema(
        fields={
            "mixture_audio": FieldSpec(
                name="mixture_audio",
                role="input",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("time",), variable_axes=("time",)),
                batching=ListBatchingSpec(),
            ),
            "sources_audio": FieldSpec(
                name="sources_audio",
                role="supervision",
                value=TensorValueSpec(dtype=torch.float32),
                shape=TensorShapeSpec(axes=("source", "time"), variable_axes=("time",)),
                batching=ListBatchingSpec(),
            ),
            "source_activity": FieldSpec(
                name="source_activity",
                role="weight",
                value=TensorValueSpec(dtype=torch.bool),
                shape=TensorShapeSpec(axes=("source",)),
                batching=StackBatchingSpec(),
            ),
        }
    )

    dataset = SourceSeparationDataset(
        feature_data=feature_data,
        schema=schema,
        audio_root_dir=str(audio_root_dir),
        target_sr=16000,
        clip_duration_seconds=1.0,
        samples_keys=samples_keys,
        target_data=target_data,
        max_num_sources=4,
    )
    dataset.setup()
    return dataset


@pytest.fixture()
def audio_root_dir(tmp_path: Path) -> Path:
    audio_root_dir = tmp_path / "data" / "raw" / "UrbanSound8K"
    audio_path = audio_root_dir / "audio" / "fold1" / "example.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_path)
    write_wav(audio_root_dir / "audio" / "fold1" / "source1.wav", num_samples=12000)
    write_wav(audio_root_dir / "audio" / "fold1" / "source2.wav", num_samples=16000)
    return audio_root_dir


def test_audio_dataset_resolves_relative_audio_path(audio_root_dir: Path) -> None:
    dataset = build_dataset(audio_root_dir, "audio/fold1/example.wav")

    sample = dataset[0]

    assert isinstance(sample, Sample)
    assert sample.fields["waveform"].shape == (16000,)


def test_audio_dataset_normalizes_legacy_absolute_audio_path(audio_root_dir: Path) -> None:
    dataset = build_dataset(
        audio_root_dir,
        "/legacy/worktree/data/raw/UrbanSound8K/audio/fold1/example.wav",
    )

    sample = dataset[0]

    assert sample.fields["waveform"].shape == (16000,)


def test_audio_dataset_raises_for_unresolvable_audio_path(audio_root_dir: Path) -> None:
    dataset = build_dataset(audio_root_dir, "audio/fold1/missing.wav")

    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_source_separation_dataset_pads_sources_and_builds_activity(audio_root_dir: Path) -> None:
    dataset = build_separation_dataset(
        audio_root_dir,
        ("audio/fold1/source1.wav", "audio/fold1/source2.wav"),
    )

    sample = dataset[0]

    assert sample.fields["mixture_audio"].shape == (16000,)
    assert sample.fields["sources_audio"].shape == (4, 16000)
    assert sample.fields["source_activity"].tolist() == [True, True, False, False]


def test_source_separation_dataset_raises_when_source_count_exceeds_limit(audio_root_dir: Path) -> None:
    dataset = build_separation_dataset(
        audio_root_dir,
        (
            "audio/fold1/source1.wav",
            "audio/fold1/source2.wav",
            "audio/fold1/example.wav",
            "audio/fold1/source1.wav",
            "audio/fold1/source2.wav",
        ),
    )

    with pytest.raises(ValueError, match="max_num_sources"):
        dataset[0]
