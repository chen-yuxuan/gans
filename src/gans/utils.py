import numpy
import torch
import random
from omegaconf import DictConfig
import os
from typing import Dict, Any


# names of hyper-parameters
PARAM_NAMES = [
    "batch_size",
    "hidden_size",
    "latent_size",
    "num_epochs",
    "g_learning_rate",
    "d_learning_rate",
    "weight_decay",
]


def seed_everything(seed: int) -> None:
    """Sets random seed anywhere randomness is involved.
    This process makes sure all the randomness-involved operations yield the
    same result under the same `seed`, so each experiment is reproducible.
    In this function, we set the same random seed for the following modules:
    `random`, `numpy` and `torch`.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_relative_path(cfg: DictConfig, start_path: str) -> None:
    """Resolves all the relative path(s) given in `config.dataset` into absolute path(s).
    This function makes our code runnable in docker as well, where using relative path has
    problem with locating dataset files in `src/../datasets`.
    Args:
        cfg: Configuration of the experiment given in a dict.
        start_path: the absolute path of the starting point, usually the running \
            script that call this function.

    Example:
        Given `cfg.dataset.root="./datasets` and
        `start_path="/netscratch/user/code/gans/main.py"`, then `cfg.dataset.root` is
        overwritten by `/netscratch/user/code/gans/datasets`.
    """
    # go from `start_path` up to the `gans` project directory (i.e. `base_path`)
    base_path = start_path
    while os.path.dirname(base_path) not in ["/", ""]:
        if base_path[-5:] == "/gans":
            break
        base_path = os.path.dirname(base_path)

    for config_column_name in ["root", "data_files"]:
        if config_column_name in cfg.dataset:
            path = cfg.dataset[config_column_name]
            # if the path is local relative
            if path[0] == ".":
                absolute_path = os.path.abspath(os.path.join(base_path, path))
                if not os.path.exists(absolute_path):
                    raise ValueError(
                        "Resolved absolute path {} does not exist, "
                        "please check your config path again".format(absolute_path)
                    )
                cfg.dataset[config_column_name] = absolute_path


def read_params_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    """Read hyperparameters from configuration.

    Args:
        cfg: Configuration of the experiment given in a dict.
    Returns:
        A dictionary containing the hyperparameters fron `cfg`.
    """
    params = {}
    for name in PARAM_NAMES:
        if name in cfg:
            params[name] = cfg[name]
    return params
