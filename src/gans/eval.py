import torch
from torch.utils.data import DataLoader

from hydra.utils import instantiate
from omegaconf import DictConfig

from .utils import seed_everything, read_params_from_cfg
from .trainers import train_gan, train_cgan


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
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    input_shape = dataset.data[0].shape  # (28, 28) for MNIST
    num_classes = len(dataset.classes)
    params = read_params_from_cfg(cfg)

    # training
    if cfg.model.upper() == "GAN":
        G = train_gan(dataloader, device=device, input_shape=input_shape, **params)
    elif cfg.model.upper() == "CGAN":
        G = train_cgan(
            dataloader,
            device=device,
            input_shape=input_shape,
            num_classes=num_classes,
            **params
        )
    else:
        raise ValueError("Unsupported model: {}".format(cfg.model))

    # validation, i.e. generate images from a given latent vector
    z = torch.randn(100, cfg.latent_size).to(device)
    if cfg.model.upper() in ["GAN"]:
        generated_images = G(z)
    elif cfg.model.upper() in ["CGAN"]:
        labels = torch.Tensor([[i] * 10 for i in range(10)])
        generated_images = G(z, labels)

    return generated_images.view(-1, 1, input_shape[-2], input_shape[-1])
