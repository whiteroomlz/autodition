from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hydra
import onnx
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.batch import Batch
from src.models.components.base import setup_modules
from src.models.components.full_models.ast import ASTAudioClassifier


class PipelineLogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_field_name: str, prediction_name: str) -> None:
        super().__init__()
        self.model = model
        self.input_field_name = input_field_name
        self.prediction_name = prediction_name

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        batch_size = model_input.shape[0]
        batch = Batch(
            sample_ids=tuple(str(index) for index in range(batch_size)),
            fields={self.input_field_name: model_input},
        )
        result = self.model(batch)
        return result.preds[self.prediction_name].value


class ASTCoreLogitsWrapper(torch.nn.Module):
    def __init__(self, classifier: torch.nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(input_values=input_values).logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export UrbanSound8K classification checkpoints to ONNX."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Lightning .ckpt path.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Hydra experiment config name, for example us8k_cnn_baseline.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output .onnx path.")
    parser.add_argument("--device", default="cpu", help="Export device. Defaults to cpu.")
    parser.add_argument("--opset", default=17, type=int, help="ONNX opset version.")
    parser.add_argument("--batch-size", default=1, type=int, help="Dummy export batch size.")
    parser.add_argument(
        "--time-steps",
        default=None,
        type=int,
        help="Dummy time length for CNN spectrogram export. Defaults to a datamodule batch shape.",
    )
    parser.add_argument(
        "--mel-bins",
        default=128,
        type=int,
        help="Mel bins for dummy spectrogram/AST feature input.",
    )
    parser.add_argument(
        "--ast-max-length",
        default=1024,
        type=int,
        help="AST input_values frame length after Hugging Face feature extraction.",
    )
    parser.add_argument(
        "--ast-attn-implementation",
        default="eager",
        help="Attention implementation used while instantiating AST for export.",
    )
    return parser.parse_args()


def compose_config(experiment: str) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return hydra.compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])


def is_ast_config(cfg: DictConfig) -> bool:
    stages = cfg.model.model.get("stages", [])
    return bool(stages) and stages[0].get("_target_") == (
        "src.models.components.full_models.ast.ASTAudioClassifier"
    )


def load_task_module(cfg: DictConfig, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    task_module = hydra.utils.instantiate(cfg.model)
    setup_modules(task_module.model)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing_keys, unexpected_keys = task_module.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict mismatch:\n"
            f"missing_keys={missing_keys}\n"
            f"unexpected_keys={unexpected_keys}"
        )

    task_module.to(device)
    task_module.eval()
    return task_module


def export_cnn(
    task_module: torch.nn.Module,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    input_field_name = cfg.data.schema.fields.mel_spectrogram.name
    prediction_name = cfg.model.prediction_name
    wrapper = PipelineLogitsWrapper(task_module.model, input_field_name, prediction_name).to(device)
    wrapper.eval()

    time_steps = args.time_steps or infer_spectrogram_time_steps(task_module, cfg, args, device)
    dummy_input = torch.randn(args.batch_size, time_steps, args.mel_bins, device=device)

    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        input_names=[input_field_name],
        output_names=[prediction_name],
        dynamic_axes={
            input_field_name: {0: "batch", 1: "time"},
            prediction_name: {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    return {
        "export_type": "cnn_pipeline_logits",
        "input_name": input_field_name,
        "input_shape": ["batch", "time", args.mel_bins],
        "output_name": prediction_name,
        "output_shape": ["batch", "num_classes"],
    }


def infer_spectrogram_time_steps(
    task_module: torch.nn.Module,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    del task_module, args, device
    clip_duration = cfg.data.get("clip_duration_seconds")
    hop_length = cfg.data.mel_spectrogram_cfg.obj.get("hop_length", 512)
    sample_rate = cfg.data.get("target_sr", 16000)
    if clip_duration is None:
        return 126
    return int(round(clip_duration * sample_rate / hop_length)) + 1


def export_ast_core(
    task_module: torch.nn.Module,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    ast_stage = task_module.model.stages[0]
    if not isinstance(ast_stage, ASTAudioClassifier):
        raise TypeError(f"Expected ASTAudioClassifier stage, got {type(ast_stage).__name__}")
    classifier = ast_stage._get_model()
    wrapper = ASTCoreLogitsWrapper(classifier).to(device)
    wrapper.eval()

    input_name = "input_values"
    output_name = cfg.model.prediction_name
    dummy_input = torch.randn(
        args.batch_size,
        args.ast_max_length,
        args.mel_bins,
        device=device,
    )

    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    return {
        "export_type": "ast_core_logits",
        "input_name": input_name,
        "input_shape": ["batch", args.ast_max_length, args.mel_bins],
        "output_name": output_name,
        "output_shape": ["batch", "num_classes"],
        "preprocessing_note": (
            "ASTAudioClassifier performs Hugging Face feature extraction outside the ONNX graph; "
            "feed normalized AST input_values with the same ASTFeatureExtractor settings."
        ),
    }


def write_metadata(output_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = compose_config(args.experiment)
    if is_ast_config(cfg):
        with open_dict(cfg.model.model.stages[0]):
            cfg.model.model.stages[0].attn_implementation = args.ast_attn_implementation
            cfg.model.model.stages[0].setup_feature_extractor = False

    device = torch.device(args.device)
    task_module = load_task_module(cfg, args.checkpoint, device)

    if is_ast_config(cfg):
        metadata = export_ast_core(task_module, cfg, args, device)
    else:
        metadata = export_cnn(task_module, cfg, args, device)

    metadata.update(
        {
            "checkpoint": str(args.checkpoint),
            "experiment": args.experiment,
            "opset": args.opset,
            "onnx_path": str(args.output),
            "config": OmegaConf.to_container(cfg, resolve=False),
        }
    )
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    write_metadata(args.output, metadata)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
