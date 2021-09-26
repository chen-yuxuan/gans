import torch
from torch.utils.data import DataLoader

from hydra.utils import instantiate
from omegaconf import DictConfig

from .utils import seed_everything
from .trainers import train_gan


def evaluate_config(cfg: DictConfig) -> torch.Tensor:
    """Evaluates the configuration by generating images.

    Args:
        cfg: Hydra-format configurationgiven in a dict.
    Returns:
        Generated images given in a `torch.Tensor` of shape `(64, 1, H, W)`.
    Raises:
        ValueError if the model provided by `cfg` is not yet supported.
    """
    seed_everything(cfg.seed)
    device = (
        torch.device("cuda", cfg.cuda_device)
        if cfg.cuda_device > -1
        else torch.device("cpu")
    )

    # prepare data
    dataset = instantiate(cfg.dataset)
    # (28, 28) for MNIST
    input_shape = dataset.data[0].shape
    num_classes = len(dataset.classes)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.params.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # training
    if cfg.model.upper() == "GAN":
        G = train_gan(
            dataloader=dataloader,
            device=device,
            input_shape=input_shape,
            **cfg.params
        )
    elif cfg.model.upper() == "CGAN":
        G = train_gan(
            dataloader=dataloader,
            device=device,
            input_shape=input_shape,
            **cfg.params
        )
    else:
        raise ValueError("Unknown model: {}".format(cfg.model))

    # validation, i.e. generate images from a given latent vector
    z = torch.randn(64, cfg.params.latent_size).to(device)
    generated_images = G(z).view(-1, 1, input_shape[-2], input_shape[-1])

    return generated_images
