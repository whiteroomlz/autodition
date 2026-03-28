import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.components.containers import (
    CategoricalFeatureInfo,
    FeatureSchema,
    FeatureTypeInfo,
    TargetSchema,
)
from src.data.components.dataset import AudioDataset
from src.data.components.raw_data import DfData


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
    waveform = (0.1 * np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate) * 32767).astype(np.int16)

    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(waveform.tobytes())


def build_dataset(audio_root_dir: Path, audio_path: str) -> AudioDataset:
    feature_data = MemoryData({"sample": {"audio_path": audio_path}})
    target_data = MemoryData({"sample": {"class_id": 3}})
    samples_keys = MemoryData({0: {"key": "sample"}})

    feature_schema = FeatureSchema(raw=FeatureTypeInfo(feature_names=["waveform"]))
    target_schema = TargetSchema(
        categorical=CategoricalFeatureInfo(
            feature_names=["class_id"],
            torch_dtype=torch.long,
            vocabularies_size=[10],
            embeddings_dim=[10],
        )
    )

    dataset = AudioDataset(
        feature_data=feature_data,
        feature_schema=feature_schema,
        mel_spectrogram=None,
        audio_root_dir=str(audio_root_dir),
        target_sr=16000,
        samples_keys=samples_keys,
        target_data=target_data,
        target_schema=target_schema,
        return_waveform_in_sample=True,
    )
    dataset.setup()
    return dataset


@pytest.fixture()
def audio_root_dir(tmp_path: Path) -> Path:
    audio_root_dir = tmp_path / "data" / "raw" / "UrbanSound8K"
    audio_path = audio_root_dir / "audio" / "fold1" / "example.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_path)
    return audio_root_dir


def test_audio_dataset_resolves_relative_audio_path(audio_root_dir: Path) -> None:
    dataset = build_dataset(audio_root_dir, "audio/fold1/example.wav")

    sample = dataset[0]

    assert sample.raw is not None
    assert sample.raw.shape == (16000,)


def test_audio_dataset_normalizes_legacy_absolute_audio_path(audio_root_dir: Path) -> None:
    dataset = build_dataset(
        audio_root_dir,
        "/legacy/worktree/data/raw/UrbanSound8K/audio/fold1/example.wav",
    )

    sample = dataset[0]

    assert sample.raw is not None
    assert sample.raw.shape == (16000,)


def test_audio_dataset_raises_for_unresolvable_audio_path(audio_root_dir: Path) -> None:
    dataset = build_dataset(audio_root_dir, "audio/fold1/missing.wav")

    with pytest.raises(FileNotFoundError):
        dataset[0]
