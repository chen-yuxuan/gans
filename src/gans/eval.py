import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from hydra.utils import instantiate
from omegaconf import DictConfig

from .utils import seed_everything, read_params_from_cfg
from .trainers import train_gan, train_cgan, train_wgan


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
    input_shape = dataset.data[0].shape  # (28, 28) for MNIST
    num_classes = len(dataset.classes)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # training
    params = read_params_from_cfg(cfg)
    if cfg.model.upper() == "GAN":
        G = train_gan(dataloader, device=device, input_shape=input_shape, **params)
    elif cfg.model.upper() == "WGAN":
        G = train_wgan(dataloader, device=device, input_shape=input_shape, **params)
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
    if cfg.model.upper() in ["GAN", "WGAN"]:
        generated_images = G(z)
    elif cfg.model.upper() in ["CGAN"]:
        labels = torch.Tensor([i for i in range(num_classes) for _ in range(10)])
        generated_images = G(z, labels)

    # save images of `num_classes` rows and 10 columns
    save_image(
        generated_images.view(-1, 1, input_shape[-2], input_shape[-1]),
        "./{}_gen_img.png".format(cfg.model),
        nrow=num_classes,
        normalize=True,
    )
