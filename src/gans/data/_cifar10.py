from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from typing import List


def cifar10(
    root: str = "./datasets/",
    train: bool = True,
    download: bool = False,
    stat_mean: List[float] = [0.4914, 0.4822, 0.4465],
    stat_std: List[float] = [0.247, 0.243, 0.261],
) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stat_mean, stat_std),
        ]
    )
    return CIFAR10(root=root, train=train, transform=transform, download=download)
