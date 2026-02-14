from itertools import chain
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter, ModelCheckpoint


class ModelCheckpointWithInit(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        filepath = self.format_checkpoint_name({})
        self._save_checkpoint(trainer, filepath)  # save checkpoint at init


# TODO: need debugging
class WeightsCallback(pl.Callback):
    def __init__(
        self,
        output_dir,
        param_names=None,
        aggr_func_names=None,
        period: Literal["epoch", "step"] = "epoch",
        every_n_period: int = 1,
    ):
        """
        param_names - list from names of module.named_parameters()
        aggr_func_names - list from keys of aggr_funcs dict w/ aggregating functions
        period - period to take param data, from [ epoch , step ]
        every_n_period - frequency to take param data
        """

        super().__init__()
        self.output_dir = output_dir
        self.param_names = param_names
        self.aggr_func_names = aggr_func_names
        self.period = period
        self.every_n_period = every_n_period

        assert self.period in ("epoch", "step"), "period is from [ epoch , step ]"

        self.param_w_aggr = {}  # weights
        self.param_u_aggr = {}  # weight updates
        self.param_g_aggr = {}  # weight gradients

        # to keep values from prev period
        self.param_w_prev_period = {}  # weights from prev period
        self.param_g_prev_period = {}  # weight gradients from prev period

        bins = 64
        self.aggr_funcs = {
            "tensor": lambda x: x,
            # raw
            "norm": torch.norm,
            "mean": torch.mean,
            "std": torch.std,
            "min": torch.min,
            "max": torch.max,
            "histogram": lambda x: torch.histc(x, bins=bins),
            # abs
            "norm(abs)": lambda x: torch.norm(torch.abs(x)),
            "mean(abs)": lambda x: torch.mean(torch.abs(x)),
            "std(abs)": lambda x: torch.std(torch.abs(x)),
            "min(abs)": lambda x: torch.min(torch.abs(x)),
            "max(abs)": lambda x: torch.max(torch.abs(x)),
            "histogram(abs)": lambda x: torch.histc(torch.abs(x), bins=bins),
            # log scale
            "norm(log_scale)": lambda x: torch.norm(
                torch.log(torch.abs(x) + torch.finfo(x.dtype).tiny)
            ),
            "mean(log_scale)": lambda x: torch.mean(
                torch.log(torch.abs(x) + torch.finfo(x.dtype).tiny)
            ),
            "std(log_scale)": lambda x: torch.std(
                torch.log(torch.abs(x) + torch.finfo(x.dtype).tiny)
            ),
            "min(log_scale)": lambda x: torch.min(
                torch.log(torch.abs(x) + torch.finfo(x.dtype).tiny)
            ),
            "max(log_scale)": lambda x: torch.max(
                torch.log(torch.abs(x) + torch.finfo(x.dtype).tiny)
            ),
            "histogram(log_scale)": lambda x: torch.histc(
                torch.log(torch.abs(x) + torch.finfo(x.dtype).tiny), bins=bins
            ),
        }

        if self.aggr_func_names is None:
            self.aggr_func_names = list(self.aggr_funcs.keys())

    def _get_param_aggr(self, pl_module):
        for name, param in pl_module.net.named_parameters():
            if param.requires_grad and name in self.param_names:
                if name in self.param_w_prev_period:
                    param_w_update = torch.sub(param.detach(), self.param_w_prev_period[name])
                else:
                    param_w_update = torch.zeros_like(param.detach())
                self.param_w_prev_period[name] = param.detach()

                # if param.grad is not None:
                #     if name in self.param_g_prev_period:
                #         param_g_curr_prev = torch.sub(param.grad.detach(), self.param_g_prev_period[name])
                #     else:
                #         param_g_curr_prev = torch.zeros_like(param.grad.detach())
                #     self.param_g_prev_period[name] = param.grad.detach()

                for aggr_func_name, aggr_func in self.aggr_funcs.items():
                    if aggr_func_name in self.aggr_func_names:
                        name_aggr = "|".join([name, aggr_func_name])

                        if name_aggr in self.param_w_aggr.keys():
                            self.param_w_aggr[name_aggr].append(aggr_func(param.detach()))
                        else:
                            self.param_w_aggr[name_aggr] = []

                        if name_aggr in self.param_u_aggr.keys():
                            self.param_u_aggr[name_aggr].append(aggr_func(param_w_update))
                        else:
                            self.param_u_aggr[name_aggr] = []

                        if param.grad is not None:  # grad is None at init
                            if name_aggr in self.param_g_aggr.keys():
                                self.param_g_aggr[name_aggr].append(aggr_func(param.grad.detach()))
                            else:
                                self.param_g_aggr[name_aggr] = []

    def on_train_start(self, trainer, pl_module):
        if self.param_names is None:
            self.param_names = [name for name, param in pl_module.net.named_parameters()]

        self._get_param_aggr(pl_module)  # after init

    def on_after_backward(self, trainer, pl_module):
        if self.period == "step" and trainer.global_step % self.every_n_period == 0:
            self._get_param_aggr(pl_module)

        if (
            self.period == "epoch"
            and trainer.current_epoch % self.every_n_period == 0
            and trainer.is_last_batch
        ):
            self._get_param_aggr(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        prefix_file_name = f"aggr_every_{self.every_n_period}_{self.period}_epoch{epoch}"

        if pl_module.global_rank == 0:
            torch.save(self.param_w_aggr, Path(self.output_dir) / f"{prefix_file_name}_weight.pt")
            torch.save(self.param_g_aggr, Path(self.output_dir) / f"{prefix_file_name}_grad.pt")
            torch.save(
                self.param_u_aggr, Path(self.output_dir) / f"{prefix_file_name}_weight_update.pt"
            )

        self.param_w_aggr = {}
        self.param_g_aggr = {}
        self.param_u_aggr = {}


class EmbedWriter(BasePredictionWriter):
    """Saves model predictions (output of predict_step)"""

    def __init__(
        self,
        output_dir: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"],
        tags: List[Optional[str]] = (),
        filename: Optional[str] = None,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.tags_postfix = ("_" if tags else "") + "-".join(tags)
        self.filename = filename

    def write_on_batch_end(
        self,
        trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        pass

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        torch.save(
            self.predictions_to_dict(predictions, batch_indices),
            Path(self.output_dir) / f"predictions{self.tags_postfix}_{trainer.global_rank}.pt",
        )

    @staticmethod
    def predictions_to_dict(predictions: Sequence[Any], batch_indices: Optional[Sequence[Any]]):
        """
        Convert predictions to dict with values: tensors, numpy arrays and lists.
        Concatenate all batches to 1 tensor.
        """

        pred_dict = {}
        for pred_index in range(len(predictions)):
            for key in predictions[pred_index].keys():
                if key not in pred_dict.keys():
                    pred_dict[key] = predictions[pred_index][key]

                else:
                    if isinstance(predictions[pred_index][key], list):
                        pred_dict[key].extend(predictions[pred_index][key])

                    elif isinstance(predictions[pred_index][key], torch.Tensor):
                        pred_dict[key] = torch.cat(
                            (pred_dict[key], predictions[pred_index][key]), dim=0
                        )

                    elif isinstance(predictions[pred_index][key], np.ndarray):
                        pred_dict[key] = np.concatenate(
                            (pred_dict[key], predictions[pred_index][key]), axis=0
                        )

                    else:
                        raise RuntimeError("Unexpected type of predictions!")

        pred_dict["batch_indices"] = torch.tensor(list(chain.from_iterable(batch_indices[0])))

        return pred_dict
