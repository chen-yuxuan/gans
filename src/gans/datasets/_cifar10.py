from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import List


def cifar10(
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
    return CIFAR10(root=root, train=train, transform=transform, download=download)
