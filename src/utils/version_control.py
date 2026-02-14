import json
import subprocess  # nosec B404
from pathlib import Path

from dvc.repo import Repo
from omegaconf import DictConfig
from ruamel.yaml import YAML

from . import pylogger

log = pylogger.RankedLogger(__name__, log_on_rank_zero_only=True)


def validate_dvc_status(root):
    repo = Repo(root)
    status = repo.status()

    if len(status) > 0:
        raise ValueError(f"Please call 'dvc commit'. You have uncommitted DVC changes: {status}")


def validate_dataset(root, cfg: DictConfig):
    dataset_dir = Path(cfg.data.raw_data.get("dataset_dir"))
    local_dvc_path = Path(f"{dataset_dir.absolute()}.dvc")
    remote_dvc_path = local_dvc_path.relative_to(root)

    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(local_dvc_path) as file:
        local_content = file.read()
    try:
        remote_content = subprocess.check_output(  # nosec B603 B607
            ["git", "show", f"origin/master:{remote_dvc_path}"], cwd=root
        ).decode("utf-8")
    except subprocess.CalledProcessError as e:
        log.error(f"Error while checking out {remote_dvc_path}: {e}")
        exit(1)

    return local_content == remote_content, remote_content


def get_dvc_hash(dvc_path: str):
    dvc_path = Path(dvc_path)

    if not dvc_path.exists():
        raise FileNotFoundError(f"File {dvc_path} not found.")

    yaml = YAML()
    with open(dvc_path) as file:
        dvc_data = yaml.load(file)

    hashes = []
    if "outs" in dvc_data:
        for out in dvc_data["outs"]:
            if "md5" in out:
                hashes.append(out["md5"])

    return "".join(sorted(hashes))
