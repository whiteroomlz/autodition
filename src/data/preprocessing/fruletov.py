from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

DEFAULT_FRULETOV_LABELS: tuple[str, ...] = (
    "car_acceleration",
    "car_braking",
    "car_horn",
    "car_idling",
    "moto_acceleration",
    "moto_idling",
    "siren_1",
    "siren_4",
    "siren_5",
    "tram",
    "tram_acceleration",
    "tram_braking",
    "tram_ring",
    "truck_acceleration",
    "truck_braking",
    "truck_horn",
    "truck_idling",
)


@dataclass(frozen=True)
class FruletovPreprocessingConfig:
    raw_dataset_dir: Path
    preprocessed_dir: Path
    chunk_duration_seconds: float = 10.0
    stride_seconds: float = 10.0
    target_sample_rate: int = 16000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_gap_seconds: float = 10.0
    audio_subdir: str = "clips"
    audio_format: str = "WAV"
    audio_subtype: str = "PCM_16"

    def validate(self) -> None:
        if self.chunk_duration_seconds <= 0:
            raise ValueError("chunk_duration_seconds must be positive")
        if self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("Split ratios must be positive")

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")


@dataclass(frozen=True)
class FruletovChunk:
    sample_id: str
    split: str
    class_name: str
    class_id: int
    source_id: str
    source_audio_path: str
    clip_audio_path: str
    chunk_index: int
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    source_sample_rate: int
    target_sample_rate: int


def preprocess_fruletov_dataset(config: FruletovPreprocessingConfig) -> Dict[str, int]:
    """Build a US8K-style preprocessed Fruletov dataset package."""

    config.validate()
    raw_dataset_dir = Path(config.raw_dataset_dir).expanduser().resolve()
    preprocessed_dir = Path(config.preprocessed_dir).expanduser().resolve()
    clips_dir = preprocessed_dir / config.audio_subdir

    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"Raw Fruletov directory not found: {raw_dataset_dir}")

    clips_dir.mkdir(parents=True, exist_ok=True)
    source_paths = tuple(sorted(raw_dataset_dir.glob("*.wav")))
    if not source_paths:
        raise FileNotFoundError(f"No .wav files found in {raw_dataset_dir}")

    features: Dict[str, Dict[str, str]] = {}
    targets: Dict[str, Dict[str, int]] = {}
    split_sample_ids = {"train": [], "val": [], "test": []}
    manifest_rows: List[Dict[str, object]] = []
    class_distribution: Dict[str, int] = {label: 0 for label in DEFAULT_FRULETOV_LABELS}

    for source_path in source_paths:
        source_chunks = build_source_chunks(source_path=source_path, config=config)
        for chunk in source_chunks:
            write_chunk_audio(chunk=chunk, raw_dataset_dir=raw_dataset_dir, preprocessed_dir=preprocessed_dir)
            features[chunk.sample_id] = {
                "audio_path": chunk.clip_audio_path,
                "source_id": chunk.source_id,
                "source_audio_path": chunk.source_audio_path,
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "duration_seconds": chunk.duration_seconds,
                "chunk_index": chunk.chunk_index,
                "split": chunk.split,
            }
            targets[chunk.sample_id] = {"class_id": chunk.class_id}
            split_sample_ids[chunk.split].append(chunk.sample_id)
            manifest_rows.append(asdict(chunk))
            class_distribution[chunk.class_name] += 1

    train_keys = build_keys(split_sample_ids["train"])
    val_keys = build_keys(split_sample_ids["val"])
    test_keys = build_keys(split_sample_ids["test"])
    manifest = pd.DataFrame(manifest_rows).sort_values(["source_id", "chunk_index"]).reset_index(drop=True)

    serialize_pickle(preprocessed_dir / "features.pkl", features)
    serialize_pickle(preprocessed_dir / "targets.pkl", targets)
    serialize_pickle(preprocessed_dir / "train_keys.pkl", train_keys)
    serialize_pickle(preprocessed_dir / "val_keys.pkl", val_keys)
    serialize_pickle(preprocessed_dir / "test_keys.pkl", test_keys)
    manifest.to_csv(preprocessed_dir / "manifest.csv", index=False)

    metadata = {
        "dataset_name": "fruletov",
        "num_classes": len(DEFAULT_FRULETOV_LABELS),
        "class_names": list(DEFAULT_FRULETOV_LABELS),
        "num_samples": len(features),
        "split_counts": {name: len(sample_ids) for name, sample_ids in split_sample_ids.items()},
        "class_distribution": class_distribution,
        "preprocessing": {
            "chunk_duration_seconds": config.chunk_duration_seconds,
            "stride_seconds": config.stride_seconds,
            "target_sample_rate": config.target_sample_rate,
            "split_gap_seconds": config.split_gap_seconds,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
    }
    (preprocessed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "num_samples": len(features),
        "train_samples": len(train_keys),
        "val_samples": len(val_keys),
        "test_samples": len(test_keys),
    }


def build_source_chunks(
    source_path: Path,
    config: FruletovPreprocessingConfig,
) -> List[FruletovChunk]:
    """Create chunk metadata for one Fruletov source recording."""

    info = sf.info(str(source_path))
    if info.frames <= 0:
        raise ValueError(f"Audio file has no frames: {source_path}")

    label_name = infer_label_name(source_path.stem)
    class_id = DEFAULT_FRULETOV_LABELS.index(label_name)
    source_id = slugify(source_path.stem)
    chunk_starts = build_chunk_starts(
        total_frames=info.frames,
        source_sample_rate=info.samplerate,
        chunk_duration_seconds=config.chunk_duration_seconds,
        stride_seconds=config.stride_seconds,
    )
    split_assignment = assign_splits(
        num_chunks=len(chunk_starts),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        gap_chunks=max(0, math.ceil(config.split_gap_seconds / config.stride_seconds)),
    )

    raw_dataset_dir = Path(config.raw_dataset_dir).expanduser().resolve()
    relative_source_path = source_path.relative_to(raw_dataset_dir).as_posix()
    chunks: List[FruletovChunk] = []
    chunk_num_frames = int(round(config.chunk_duration_seconds * info.samplerate))

    for chunk_index, start_frame in enumerate(chunk_starts):
        split = split_assignment[chunk_index]
        if split is None:
            continue
        end_frame = min(start_frame + chunk_num_frames, info.frames)
        sample_id = f"{source_id}__chunk_{chunk_index:05d}"
        clip_relative_path = f"{config.audio_subdir}/{sample_id}.wav"
        chunks.append(
            FruletovChunk(
                sample_id=sample_id,
                split=split,
                class_name=label_name,
                class_id=class_id,
                source_id=source_id,
                source_audio_path=relative_source_path,
                clip_audio_path=clip_relative_path,
                chunk_index=chunk_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_seconds=start_frame / info.samplerate,
                end_seconds=end_frame / info.samplerate,
                duration_seconds=(end_frame - start_frame) / info.samplerate,
                source_sample_rate=info.samplerate,
                target_sample_rate=config.target_sample_rate,
            )
        )

    return chunks


def build_chunk_starts(
    total_frames: int,
    source_sample_rate: int,
    chunk_duration_seconds: float,
    stride_seconds: float,
) -> List[int]:
    """Generate deterministic chunk starts, keeping only full chunks when possible."""

    chunk_frames = int(round(chunk_duration_seconds * source_sample_rate))
    stride_frames = int(round(stride_seconds * source_sample_rate))
    if chunk_frames <= 0 or stride_frames <= 0:
        raise ValueError("chunk_frames and stride_frames must be positive")

    if total_frames <= chunk_frames:
        return [0]

    max_start = total_frames - chunk_frames
    return list(range(0, max_start + 1, stride_frames))


def assign_splits(
    num_chunks: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    gap_chunks: int,
) -> List[str | None]:
    """Assign contiguous train/val/test blocks with guard gaps between them."""

    if num_chunks < 3:
        raise ValueError("At least 3 chunks are required to build train/val/test splits")

    total_gap = gap_chunks * 2
    if num_chunks <= total_gap + 2:
        raise ValueError(
            f"Not enough chunks ({num_chunks}) for requested split gap ({gap_chunks} chunks)"
        )

    usable_chunks = num_chunks - total_gap
    train_count = max(1, int(round(usable_chunks * train_ratio)))
    val_count = max(1, int(round(usable_chunks * val_ratio)))
    test_count = usable_chunks - train_count - val_count

    if test_count < 1:
        deficit = 1 - test_count
        if train_count >= val_count and train_count > 1:
            train_count -= deficit
        elif val_count > 1:
            val_count -= deficit
        test_count = 1

    total_assigned = train_count + val_count + test_count
    if total_assigned != usable_chunks:
        train_count += usable_chunks - total_assigned

    train_end = train_count
    val_start = train_end + gap_chunks
    val_end = val_start + val_count
    test_start = val_end + gap_chunks
    test_end = test_start + test_count

    assignments: List[str | None] = [None] * num_chunks
    for chunk_index in range(0, train_end):
        assignments[chunk_index] = "train"
    for chunk_index in range(val_start, val_end):
        assignments[chunk_index] = "val"
    for chunk_index in range(test_start, test_end):
        assignments[chunk_index] = "test"

    if assignments.count("train") != train_count:
        raise RuntimeError("Failed to assign the expected number of train chunks")
    if assignments.count("val") != val_count:
        raise RuntimeError("Failed to assign the expected number of val chunks")
    if assignments.count("test") != test_count:
        raise RuntimeError("Failed to assign the expected number of test chunks")

    return assignments


def write_chunk_audio(
    chunk: FruletovChunk,
    raw_dataset_dir: Path,
    preprocessed_dir: Path,
) -> None:
    """Read a source segment, convert it to mono/target sample rate, and save a clip file."""

    source_path = raw_dataset_dir / chunk.source_audio_path
    clip_path = preprocessed_dir / chunk.clip_audio_path
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    waveform, source_sample_rate = sf.read(
        str(source_path),
        start=chunk.start_frame,
        stop=chunk.end_frame,
        dtype="float32",
        always_2d=True,
    )
    waveform_tensor = torch.from_numpy(waveform.T)
    waveform_tensor = waveform_tensor.mean(dim=0, keepdim=True)

    if source_sample_rate != chunk.target_sample_rate:
        waveform_tensor = audio_functional.resample(
            waveform_tensor,
            orig_freq=source_sample_rate,
            new_freq=chunk.target_sample_rate,
        )

    sf.write(
        file=str(clip_path),
        data=waveform_tensor.squeeze(0).cpu().numpy(),
        samplerate=chunk.target_sample_rate,
        format="WAV",
        subtype="PCM_16",
    )


def build_keys(sample_ids: Sequence[str]) -> Dict[int, Dict[str, str]]:
    return {index: {"key": sample_id} for index, sample_id in enumerate(sample_ids)}


def infer_label_name(source_stem: str) -> str:
    normalized = slugify(source_stem.replace("_full", "").replace("full", ""))
    candidates = (
        normalized,
        normalized.replace("_dataset_nov_2021", ""),
        normalized.replace("_full", ""),
    )

    for candidate in candidates:
        if candidate in DEFAULT_FRULETOV_LABELS:
            return candidate

    raise ValueError(f"Cannot infer Fruletov label from source name '{source_stem}'")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def serialize_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def iter_fruletov_source_paths(raw_dataset_dir: Path) -> Iterable[Path]:
    return sorted(raw_dataset_dir.glob("*.wav"))
