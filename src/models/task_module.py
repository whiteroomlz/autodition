from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric

from src.data.components.batch import Batch
from src.models.components.base import Model, ModelResult, setup_modules
from src.models.components.metrics import MetricSuite
from src.models.components.objectives import ObjectiveComposer


class TaskModule(LightningModule):
    def __init__(
        self,
        model: Model,
        objectives: ObjectiveComposer,
        metrics: MetricSuite,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        compile: bool,
        prediction_name: Optional[str] = None,
        monitor_metric: Optional[str] = None,
        monitor_metric_mode: Literal["min", "max"] = "max",
        warmup_type: Optional[Literal["linear", "cosine"]] = None,
        warmup_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["model", "objectives", "metrics", "optimizer", "scheduler"],
        )

        self.prediction_name = prediction_name
        self.model = model
        self.objectives = objectives
        self.train_metric_suite = metrics.clone()
        self.val_metric_suite = metrics.clone()
        self.test_metric_suite = metrics.clone()
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.base_lr = getattr(optimizer, "keywords", {}).get("lr")

        self.monitor_metric_tracker = (
            MaxMetric() if monitor_metric_mode == "max" else MinMetric()
        )

    def setup(self, stage: str) -> None:
        setup_modules(self.model)

        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(self, batch: Batch) -> ModelResult:
        return self.model(batch)

    def on_train_start(self) -> None:
        self.monitor_metric_tracker.reset()

    def model_step(self, batch: Batch, split: str) -> tuple[torch.Tensor, ModelResult, Dict[str, torch.Tensor]]:
        result = self.forward(batch)
        step = self.trainer.global_step if getattr(self, "_trainer", None) is not None else 0
        total_loss, term_losses = self.objectives(batch=batch, result=result, step=step)
        self._metric_suite(split).update(batch=batch, result=result)
        return total_loss, result, term_losses

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, _, term_losses = self.model_step(batch, split="train")
        self._log_losses("train", loss, term_losses, on_step=True)
        self._log_metrics("train")
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, _, term_losses = self.model_step(batch, split="val")
        self._log_losses("val", loss, term_losses, on_step=False)
        self._log_metrics("val")
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, _, term_losses = self.model_step(batch, split="test")
        self._log_losses("test", loss, term_losses, on_step=False)
        self._log_metrics("test")
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking or not self.hparams.monitor_metric:
            return

        monitor_value = self.trainer.callback_metrics.get(self.hparams.monitor_metric)
        if monitor_value is None:
            return

        self.monitor_metric_tracker.update(monitor_value)
        self.log(
            f"{self.hparams.monitor_metric}_best",
            self.monitor_metric_tracker.compute(),
            prog_bar=True,
            sync_dist=True,
        )

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

    def _get_warmup_lr_scale(self, warmup_steps: int) -> float:
        if self.hparams.warmup_type == "linear":
            return min(1.0, float(self.trainer.global_step + 1) / warmup_steps)
        if self.hparams.warmup_type == "cosine":
            import numpy as np

            return 0.5 * (
                1 + np.cos(np.pi * (1 - float(self.trainer.global_step + 1) / warmup_steps))
            )
        return 1.0

    def _log_losses(
        self,
        split: str,
        total_loss: torch.Tensor,
        term_losses: Dict[str, torch.Tensor],
        *,
        on_step: bool,
    ) -> None:
        self.log(
            f"{split}/loss",
            total_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=(split != "test"),
            sync_dist=(split != "train"),
        )
        for name, loss in term_losses.items():
            self.log(
                f"{split}/loss/{name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=(split != "train"),
            )

    def _log_metrics(self, split: str) -> None:
        for name, metric in self._metric_suite(split).metric_objects().items():
            self.log(
                f"{split}/{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=split != "test",
                sync_dist=(split != "train"),
            )

    def _metric_suite(self, split: str) -> MetricSuite:
        if split == "train":
            return self.train_metric_suite
        if split == "val":
            return self.val_metric_suite
        if split == "test":
            return self.test_metric_suite
        raise KeyError(f"Unsupported split '{split}'")
