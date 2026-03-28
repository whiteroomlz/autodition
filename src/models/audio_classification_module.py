from abc import ABC
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src.data.components.batch import Batch
from src.models.components.base import (
    Model,
    ModelInput,
    ModelOutput,
    ModelOutputForClassification,
    setup_modules,
)
from src.models.components.batch_adapter import BatchToModelInputAdapter
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, log_on_rank_zero_only=True)


class LightningNet(LightningModule, ABC):
    net: Model

    def forward(self, x: ModelInput) -> ModelOutput:
        return self.net(x)


class AudioClassificationModule(LightningNet):
    def __init__(
        self,
        net: Model,
        input_adapter: BatchToModelInputAdapter,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        compile: bool,
        num_classes: int,
        target_field: str,
        monitor_metric: Optional[str] = None,
        warmup_type: Optional[Literal["linear", "cosine"]] = None,
        warmup_rate: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "input_adapter", "optimizer", "scheduler"],
        )

        self.net = net
        self.input_adapter = input_adapter
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.base_lr = getattr(optimizer, "keywords", {}).get("lr")
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def setup(self, stage: str) -> None:
        setup_modules(self.net)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, x: ModelInput) -> ModelOutputForClassification:
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_acc_best.reset()

    def model_step(
        self,
        batch: Batch,
    ) -> Tuple[torch.Tensor, ModelOutputForClassification, torch.Tensor]:
        model_input = self.input_adapter(batch)
        if self.hparams.target_field not in batch.fields:
            raise KeyError(f"Batch does not contain target field '{self.hparams.target_field}'")

        targets = torch.as_tensor(batch.fields[self.hparams.target_field]).long().view(-1)
        output = self.forward(model_input)
        loss = self.criterion(output.logits, targets)
        return loss, output, targets

    def training_step(self, batch: Batch, batch_idx: int):
        loss, output, targets = self.model_step(batch)
        preds = output.logits

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        loss, output, targets = self.model_step(batch)
        preds = output.logits

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Batch, batch_idx: int):
        loss, output, targets = self.model_step(batch)
        preds = output.logits

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer_factory(params=self.trainer.model.parameters())

        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor_metric or "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.hparams.warmup_rate and self.hparams.warmup_type:
            lr_scale = 1.0
            warmup_steps = int(self.hparams.warmup_rate * self.trainer.estimated_stepping_batches)

            if self.trainer.global_step < warmup_steps:
                lr_scale = self._get_warmup_lr_scale(warmup_steps)

            base_lr = self.base_lr or optimizer.param_groups[0]["lr"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scale * base_lr

        optimizer.step(closure=optimizer_closure)

    def _get_warmup_lr_scale(self, warmup_steps):
        if self.hparams.warmup_type == "linear":
            return min(1.0, float(self.trainer.global_step + 1) / warmup_steps)
        if self.hparams.warmup_type == "cosine":
            import numpy as np

            return 0.5 * (
                1 + np.cos(np.pi * (1 - float(self.trainer.global_step + 1) / warmup_steps))
            )
        return 1.0
