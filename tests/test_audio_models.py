from pathlib import Path

import hydra
import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.data.components.collate import ModelBatch
from src.models.audio_classification_module import AudioClassificationModule
from src.models.components.base import SequentialModelInput
from src.models.components.full_models.ast import ASTAudioClassifier
from src.models.components.input_blocks.spectrogram_cnn import SpectrogramCNNEncoder


def test_spectrogram_cnn_encoder_output_shape() -> None:
    encoder = SpectrogramCNNEncoder(
        channels=(16, 32),
        embedding_dim=64,
        kernel_size=3,
        dropout=0.1,
    )
    model_input = SequentialModelInput(
        numerical=torch.randn(4, 128, 128),
        padding_mask=torch.ones(4, 128, dtype=torch.bool),
    )

    output = encoder(model_input)

    assert output.hidden_state.shape == (4, 64)


def test_audio_classification_module_model_step_with_ast_net() -> None:
    net = ASTAudioClassifier(
        num_classes=10,
        model_name="unused",
        load_pretrained=False,
        model_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
        },
    )
    module = AudioClassificationModule(
        net=net,
        optimizer=hydra.utils.get_object("torch.optim.AdamW"),
        scheduler=None,
        compile=False,
        num_classes=10,
        target_name="class_id",
    )
    module.setup("fit")

    batch = ModelBatch(
        sample_ids=("a", "b"),
        raw=(torch.randn(16000), torch.randn(16000)),
        numerical=None,
        categorical=None,
        targets={"class_id": torch.tensor([[0], [1]])},
        padding_mask=None,
    )

    loss, output, targets = module.model_step(batch)

    assert loss.ndim == 0
    assert output.logits.shape == (2, 10)
    assert targets.shape == (2,)


def test_ast_audio_classifier_requires_setup() -> None:
    net = ASTAudioClassifier(
        num_classes=10,
        model_name="unused",
        load_pretrained=False,
        model_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
        },
    )

    with pytest.raises(RuntimeError, match="Setup method call required"):
        net(SequentialModelInput(raw=(torch.randn(16000),)))


@pytest.mark.parametrize(
    ("experiment_name", "extra_overrides"),
    [
        ("us8k_cnn_baseline", []),
        ("us8k_ast_finetune", []),
    ],
)
def test_us8k_experiment_configs_instantiate(
    experiment_name: str,
    extra_overrides: list[str],
    tmp_path: Path,
) -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[f"experiment={experiment_name}", *extra_overrides],
        )

    with open_dict(cfg):
        cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = None

    HydraConfig().set_config(cfg)

    hydra.utils.instantiate(cfg.data)
    hydra.utils.instantiate(cfg.model)
    hydra.utils.instantiate(cfg.trainer)
    GlobalHydra.instance().clear()
