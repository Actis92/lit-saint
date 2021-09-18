from typing import Tuple

import torch
import numpy as np


def mixup(x: torch.Tensor, random_index: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """It apply mixup augmentation, making a weighted average between a tensor
        and some random element of the tensor taking random rows
    """
    return lam * x + (1 - lam) * x[random_index, :]


def cutmix(x: torch.Tensor, random_index: torch.Tensor, noise_lambda: float = 0.1) -> torch.Tensor:
    """Define how apply cutmix to a tensor"""
    x_binary_mask = torch.from_numpy(np.random.choice(2, size=x.shape, p=[noise_lambda, 1 - noise_lambda]))
    x_random = x[random_index, :]
    x_noised = x.clone().detach()
    x_noised[x_binary_mask == 0] = x_random[x_binary_mask == 0]
    return x_noised


def get_random_index(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    return index
