import functools
import os

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata


def register_torch_safe_globals() -> None:
    """Allow trusted local Lightning checkpoints to load under Torch 2.6 defaults."""

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    add_safe_globals(
        [
            functools.partial,
            DictConfig,
            ListConfig,
            ContainerMetadata,
            torch.optim.AdamW,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ]
    )
