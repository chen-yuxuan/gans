import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

from gans.eval import evaluate_config
from gans.utils import resolve_relative_path


@hydra.main(config_name="config", config_path="config")
def evaluate(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg, start_path=os.path.abspath(__file__))
    print(OmegaConf.to_yaml(cfg))

    generated_images = evaluate_config(cfg)

    # save images
    save_image(
        generated_images, "./{}_gen_img.png".format(cfg.model), nrow=8, normalize=True
    )


if __name__ == "__main__":
    evaluate()
