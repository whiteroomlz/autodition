from __future__ import annotations

import argparse

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.preprocessing.fruletov import (  # noqa: E402
    FruletovPreprocessingConfig,
    preprocess_fruletov_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Fruletov into US8K-style artifacts.")
    parser.add_argument(
        "--raw-dir",
        default=str(root / "data" / "raw" / "Fruletov" / "Dataset Nov 2021"),
        help="Directory with raw Fruletov wav files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "data" / "preprocessed" / "fruletov"),
        help="Directory where preprocessed artifacts will be written.",
    )
    parser.add_argument("--chunk-duration", type=float, default=10.0, help="Chunk duration in seconds.")
    parser.add_argument("--stride", type=float, default=10.0, help="Stride between chunk starts in seconds.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Output sample rate.")
    parser.add_argument("--split-gap", type=float, default=10.0, help="Gap between split blocks in seconds.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = preprocess_fruletov_dataset(
        FruletovPreprocessingConfig(
            raw_dataset_dir=args.raw_dir,
            preprocessed_dir=args.output_dir,
            chunk_duration_seconds=args.chunk_duration,
            stride_seconds=args.stride,
            target_sample_rate=args.target_sr,
            split_gap_seconds=args.split_gap,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    )
    print("Fruletov preprocessing complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
