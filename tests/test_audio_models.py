from pathlib import Path

import hydra
import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from torchmetrics.classification import MulticlassAccuracy

from src.data.components.batch import Batch
from src.models.components.base import BlockModel, ModelContext, PipelineModel, Ref, setup_modules
from src.models.components.full_models.ast import ASTAudioClassifier
from src.models.components.full_models.sudormrf import SuDORMRFSeparator
from src.models.components.full_models.tfgridnet import TFGridNetSeparator
from src.models.components.input_blocks.spectrogram_cnn import SpectrogramCNNEncoder
from src.models.components.metrics import MetricSuite, SupervisedMetricTerm
from src.models.components.objectives import (
    CrossEntropyCriterion,
    ObjectiveComposer,
    SupervisedLossTerm,
)
from src.models.components.output_blocks.classifiers.flat_linear import FlatLinearClassifier
from src.models.task_module import TaskModule


def test_pipeline_model_with_spectrogram_cnn_outputs_class_logits() -> None:
    model = PipelineModel(
        stages=[
            BlockModel(
                inputs=[Ref(source="field", name="mel_spectrogram")],
                output_name="class_logits",
                store="pred",
                input_block=SpectrogramCNNEncoder(
                    channels=(16, 32),
                    embedding_dim=64,
                    kernel_size=3,
                    dropout=0.1,
                ),
                hidden_blocks=[],
                output_block=FlatLinearClassifier(emb_dim=64, num_classes=10),
            )
        ]
    )
    setup_modules(model)

    batch = Batch(
        sample_ids=("a", "b", "c", "d"),
        fields={"mel_spectrogram": torch.randn(4, 128, 128)},
        masks={"mel_spectrogram": torch.ones(4, 128, dtype=torch.bool)},
        meta={},
    )
    result = model(batch)

    assert result.preds["class_logits"].value.shape == (4, 10)


def test_task_module_model_step_with_ast_stage() -> None:
    model = PipelineModel(
        stages=[
            ASTAudioClassifier(
                inputs=[Ref(source="field", name="waveform")],
                output_name="class_logits",
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
        ]
    )
    module = TaskModule(
        model=model,
        objectives=ObjectiveComposer(
            terms=[
                SupervisedLossTerm(
                    name="classification",
                    prediction_ref=Ref(source="pred", name="class_logits"),
                    target_ref=Ref(source="field", name="class_id"),
                    criterion=CrossEntropyCriterion(),
                )
            ]
        ),
        metrics=MetricSuite(
            terms=[
                SupervisedMetricTerm(
                    name="acc",
                    prediction_ref=Ref(source="pred", name="class_logits"),
                    target_ref=Ref(source="field", name="class_id"),
                    metric=MulticlassAccuracy(num_classes=10, average="macro"),
                )
            ]
        ),
        optimizer=hydra.utils.get_object("torch.optim.AdamW"),
        scheduler=None,
        compile=False,
        monitor_metric="val/acc",
    )
    module.setup("fit")

    batch = Batch(
        sample_ids=("a", "b"),
        fields={
            "waveform": torch.randn(2, 16000),
            "class_id": torch.tensor([0, 1]),
        },
        masks={"waveform": torch.ones(2, 16000, dtype=torch.bool)},
        meta={},
    )

    loss, result, term_losses = module.model_step(batch, split="train")

    assert loss.ndim == 0
    assert result.preds["class_logits"].value.shape == (2, 10)
    assert term_losses["classification"].ndim == 0


def test_sudormrf_separator_outputs_sources_audio() -> None:
    model = PipelineModel(
        stages=[
            SuDORMRFSeparator(
                inputs=[Ref(source="field", name="mixture_audio")],
                output_name="sources_audio",
                num_sources=4,
                out_channels=64,
                bottleneck_channels=128,
                num_blocks=2,
                upsampling_depth=3,
                enc_kernel_size=9,
                enc_num_basis=64,
                enforce_mixture_consistency=True,
            )
        ]
    )
    setup_modules(model)

    batch = Batch(
        sample_ids=("a", "b"),
        fields={"mixture_audio": torch.randn(2, 8000)},
        masks={"mixture_audio": torch.ones(2, 8000, dtype=torch.bool)},
        meta={},
    )
    result = model(batch)
    prediction = result.preds["sources_audio"].value

    assert prediction.shape == (2, 4, 8000)
    assert torch.allclose(prediction.sum(dim=1), batch.fields["mixture_audio"], atol=1e-5)


def test_tfgridnet_separator_outputs_sources_audio() -> None:
    model = PipelineModel(
        stages=[
            TFGridNetSeparator(
                inputs=[Ref(source="field", name="mixture_audio")],
                output_name="sources_audio",
                num_sources=4,
                n_fft=128,
                hop_length=32,
                num_layers=1,
                lstm_hidden_units=32,
                emb_dim=16,
                emb_kernel_size=2,
                emb_hop_size=1,
                num_heads=4,
                approx_qk_dim=64,
                enforce_mixture_consistency=True,
            )
        ]
    )
    setup_modules(model)

    batch = Batch(
        sample_ids=("a", "b"),
        fields={"mixture_audio": torch.randn(2, 2048)},
        masks={"mixture_audio": torch.ones(2, 2048, dtype=torch.bool)},
        meta={},
    )
    result = model(batch)
    prediction = result.preds["sources_audio"].value

    assert prediction.shape == (2, 4, 2048)
    assert torch.allclose(prediction.sum(dim=1), batch.fields["mixture_audio"], atol=1e-5)


def test_ast_audio_classifier_requires_setup() -> None:
    stage = ASTAudioClassifier(
        inputs=[Ref(source="field", name="waveform")],
        output_name="class_logits",
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
    context = Batch(
        sample_ids=("a",),
        fields={"waveform": torch.randn(1, 16000)},
        masks={"waveform": torch.ones(1, 16000, dtype=torch.bool)},
        meta={},
    )

    with pytest.raises(RuntimeError, match="setup"):
        stage(ModelContext.from_batch(context))


@pytest.mark.parametrize(
    "experiment_name",
    [
        "us8k_cnn_baseline",
        "us8k_ast_finetune",
        "fuss_sudormrf_baseline",
        "fuss_tfgridnet_small",
    ],
)
def test_us8k_experiment_configs_instantiate(
    experiment_name: str,
    tmp_path: Path,
) -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[f"experiment={experiment_name}"],
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
