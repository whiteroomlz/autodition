import hydra
import rootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger
from src.utils.version_control import get_dvc_hash

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


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
