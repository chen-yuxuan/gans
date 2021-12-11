from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

from typing import List


def mnist(
    root: str = "./datasets/",
    train: bool = True,
    download: bool = False,
    stat_mean: List[float] = [0.5],
    stat_std: List[float] = [0.5],
) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stat_mean, stat_std),
        ]
    )
    return MNIST(root=root, train=train, transform=transform, download=download)
