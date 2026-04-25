from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import rootutils
import soundfile as sf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.data.preprocessing.fruletov import (
    FruletovPreprocessingConfig,
    preprocess_fruletov_dataset,
)
from src.train import train

PROJECT_ROOT = rootutils.find_root(indicator=".project-root")


def write_stereo_wav(
    path: Path,
    sample_rate: int = 8000,
    duration_seconds: float = 30.0,
    frequency_hz: float = 440.0,
) -> None:
    num_samples = int(sample_rate * duration_seconds)
    time_axis = np.arange(num_samples, dtype=np.float32) / sample_rate
    left = 0.1 * np.sin(2 * np.pi * frequency_hz * time_axis)
    right = 0.1 * np.sin(2 * np.pi * (frequency_hz * 1.5) * time_axis)
    waveform = np.stack([left, right], axis=1)
    sf.write(path, waveform, sample_rate)


@pytest.fixture()
def fruletov_source_dir(tmp_path: Path) -> Path:
    source_dir = tmp_path / "raw" / "Fruletov" / "Dataset Nov 2021"
    source_dir.mkdir(parents=True, exist_ok=True)

    write_stereo_wav(source_dir / "Car acceleration Full.wav", frequency_hz=220.0)
    write_stereo_wav(source_dir / "Siren 1 Full.wav", frequency_hz=330.0)
    write_stereo_wav(source_dir / "Truck Horn Full.wav", frequency_hz=440.0)
    return source_dir


@pytest.fixture()
def fruletov_preprocessed_dir(tmp_path: Path, fruletov_source_dir: Path) -> Path:
    preprocessed_dir = tmp_path / "preprocessed" / "fruletov"
    preprocess_fruletov_dataset(
        FruletovPreprocessingConfig(
            raw_dataset_dir=fruletov_source_dir,
            preprocessed_dir=preprocessed_dir,
            chunk_duration_seconds=10.0,
            stride_seconds=10.0,
            target_sample_rate=16000,
            split_gap_seconds=0.0,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
    )
    return preprocessed_dir


def test_preprocess_fruletov_dataset_writes_us8k_style_artifacts(
    fruletov_preprocessed_dir: Path,
) -> None:
    expected_files = {
        "features.pkl",
        "targets.pkl",
        "train_keys.pkl",
        "val_keys.pkl",
        "test_keys.pkl",
        "manifest.csv",
        "metadata.json",
    }
    assert expected_files.issubset({path.name for path in fruletov_preprocessed_dir.iterdir()})

    with (fruletov_preprocessed_dir / "features.pkl").open("rb") as stream:
        features = pickle.load(stream)
    with (fruletov_preprocessed_dir / "targets.pkl").open("rb") as stream:
        targets = pickle.load(stream)
    metadata = json.loads((fruletov_preprocessed_dir / "metadata.json").read_text(encoding="utf-8"))

    assert len(features) == 9
    assert len(targets) == 9
    assert metadata["split_counts"] == {"train": 3, "val": 3, "test": 3}

    first_sample_id = next(iter(features))
    first_feature = features[first_sample_id]
    first_target = targets[first_sample_id]
    clip_path = fruletov_preprocessed_dir / first_feature["audio_path"]

    assert clip_path.exists()
    assert first_target["class_id"] in {0, 6, 15}

    clip_info = sf.info(str(clip_path))
    assert clip_info.samplerate == 16000
    assert clip_info.channels == 1
    assert first_feature["split"] in {"train", "val", "test"}


def test_fruletov_ast_training_smoke(fruletov_preprocessed_dir: Path) -> None:
    try:
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=["experiment=fruletov_ast_finetune"],
            )

        with open_dict(cfg):
            cfg.paths.root_dir = str(PROJECT_ROOT)
            cfg.paths.preprocessed_data_dir = str(fruletov_preprocessed_dir.parent)
            cfg.paths.data_dir = str(fruletov_preprocessed_dir.parent)
            cfg.trainer.fast_dev_run = True
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.logger = None
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.model.model.stages[0].load_pretrained = False
            cfg.model.model.stages[0].model_name = "ast-smoke"
            cfg.model.model.stages[0].model_config.hidden_size = 64
            cfg.model.model.stages[0].model_config.num_hidden_layers = 2
            cfg.model.model.stages[0].model_config.num_attention_heads = 4
            cfg.model.model.stages[0].model_config.intermediate_size = 128

        HydraConfig().set_config(cfg)
        metric_dict, _ = train(cfg)
        assert "train/loss" in metric_dict
    finally:
        GlobalHydra.instance().clear()
