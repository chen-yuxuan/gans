import os
import random
from typing import Dict, Any
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import numpy
import torch


# names of hyper-parameters
_HYPERPARAM_NAMES = [
    "batch_size",
    "hidden_size",
    "latent_size",
    "num_epochs",
    "g_lr",
    "d_lr",
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


def resolve_relative_path(cfg: DictConfig) -> None:
    """Resolves all the relative path(s) given in `config.dataset` into absolute path(s).
    This function makes our code runnable in docker as well, where using relative path has
    problem with locating dataset files in `src/../datasets`.
    Args:
        cfg: Configuration of the experiment given in a dict.

    Example:
        Given `cfg.dataset.root="./datasets` and we call from
        "/netscratch/user/code/gans/main.py", then `cfg.dataset.root` is
        overwritten by `/netscratch/user/code/gans/datasets`.
    """
    for config_column_name in ["root", "data_files"]:
        if config_column_name in cfg.dataset:
            cfg.dataset[config_column_name] = to_absolute_path(
                cfg.dataset[config_column_name]
            )


def read_hyperparams_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    """Read hyperparameters from configuration.

    Args:
        cfg: Configuration of the experiment given in a dict.
    Returns:
        A dictionary containing the hyperparameters fron `cfg`.
    """
    params = {}
    for name in _HYPERPARAM_NAMES:
        if name in cfg:
            params[name] = cfg[name]
    return params
