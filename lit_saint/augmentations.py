from typing import Tuple

import torch
import numpy as np


def mixup_data(x1: torch.Tensor, x2: torch.Tensor, lam: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """It apply mixup augmentation, making a weighted average between a tensor
        and some random element of the tensor
    """
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size)

    mixed_x1: torch.Tensor = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2: torch.Tensor = lam * x2 + (1 - lam) * x2[index, :]

    return mixed_x1, mixed_x2


def add_noise(x1: torch.Tensor, x2: torch.Tensor, noise_lambda: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """It apply cutmix augmentation to 2 tensors, replace some element of the tensors with some random element"""
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size)
    x1_noised = _add_noise_single_tensor(x1, index, noise_lambda)
    x2_noised = _add_noise_single_tensor(x2, index, noise_lambda)
    return x1_noised, x2_noised


def _add_noise_single_tensor(x: torch.Tensor, index: torch.Tensor, noise_lambda: float = 0.1) -> torch.Tensor:
    """Define how apply cutmix to a tensor"""
    x_binary_mask = torch.from_numpy(np.random.choice(2, size=x.shape, p=[noise_lambda, 1 - noise_lambda]))
    x_random = x[index, :]
    x_noised = x.clone().detach()
    x_noised[x_binary_mask == 0] = x_random[x_binary_mask == 0]
    return x_noised
