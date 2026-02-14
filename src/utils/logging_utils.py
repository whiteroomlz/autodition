import logging
from typing import Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf

from . import pylogger

log = pylogger.RankedLogger(__name__, log_on_rank_zero_only=True)


def setup_debug_logger(logger: logging.Logger):
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: Dict) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally, saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams["model_architecture"] = log_params_from_omegaconf_dict(
        OmegaConf.create(cfg)["model"]["net"]
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")

    hparams["total_train_batch_size"] = get_effective_batch_size_from_cfg(cfg)

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def _explore_recursive(logger, parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            _explore_recursive(logger, f"{parent_name}/{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            _explore_recursive(logger, f"{parent_name}/{i}", v)
    else:
        logger[parent_name] = element


def log_params_from_omegaconf_dict(config):
    """Save config file to logger parameters :param config: configuration file :return:"""
    res = {}
    for param_name, element in config.items():
        _explore_recursive(res, param_name, element)
    return res


def get_effective_batch_size_from_cfg(cfg: dict):
    devices = cfg["trainer"].get("devices", 1)

    if isinstance(devices, ListConfig):
        devices_count = len(devices)
    else:
        devices_count = devices

    effective_batch_size = (
        devices_count
        * cfg["trainer"].get("accumulate_grad_batches", 1)
        * cfg["data"].get("train_batch_size", 1)
        * cfg["trainer"].get("num_nodes", 1)
    )
    return effective_batch_size
