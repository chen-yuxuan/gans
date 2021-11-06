import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from gans.eval import evaluate_config
from gans.utils import resolve_relative_path


logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="configs")
def evaluate(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg)
    print(OmegaConf.to_yaml(cfg))

    evaluate_config(cfg)


if __name__ == "__main__":
    evaluate()
