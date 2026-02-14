from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import pytorch_lightning as pl
import rootutils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.version_control import (
    get_dvc_hash,
    validate_dataset,
    validate_dvc_status,
)

log = RankedLogger(__name__, log_on_rank_zero_only=True)

try:
    OmegaConf.register_new_resolver("eval", eval)  # example: ${eval:${model.net.emb_dim} * 2}
except ValueError:
    log.warning('Cannot register "eval" resolver')

try:
    OmegaConf.register_new_resolver(
        "get_dvc_hash", get_dvc_hash
    )  # example: ${get_dvc_hash:<path>}
except ValueError:
    log.warning('Cannot register "get_dvc_hash" resolver')


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

        for logger_instance in logger:
            config_path = Path(cfg.paths.output_dir, "config.yaml")
            config_not_resolved_path = Path(cfg.paths.output_dir, "config_not_resolved.yaml")

            if isinstance(logger_instance, pl.loggers.mlflow.MLFlowLogger):
                logger_instance.experiment.log_artifact(logger_instance.run_id, config_path)
                logger_instance.experiment.log_artifact(
                    logger_instance.run_id, config_not_resolved_path
                )

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    if cfg.get("debug", None) is None:
        log.info("Running data validation...")
        validate_dvc_status(root)
        is_valid, remote_dvc_content = validate_dataset(root=root, cfg=cfg)
        if not is_valid:
            raise ValueError(
                "Config parameters specify the use of the master dataset; current dataset does not match the master."
            )

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
