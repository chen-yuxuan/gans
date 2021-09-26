from gans.utils import seed_everything
import random
import numpy as np
import torch


def test_seed_everything():
    seed_everything(1234)
    x_random = random.randint(0, 10000)
    x_np_random = np.random.randint(0, 10000)
    x_torch_random = torch.randint(0, 10000, (1,))

    seed_everything(1234)
    y_random = random.randint(0, 10000)
    y_np_random = np.random.randint(0, 10000)
    y_torch_random = torch.randint(0, 10000, (1,))

    assert x_random == y_random
    assert x_np_random == y_np_random
    assert x_torch_random == y_torch_random
