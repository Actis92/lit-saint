import torch
import numpy as np


def mixup(x: torch.Tensor, random_index: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """It apply mixup augmentation, making a weighted average between a tensor
    and some random element of the tensor taking random rows

    :param x: Tensor on which apply the mixup augmentation
    :param random_index: list of indices used to permute the tensor
    :param lam: weight in the linear combination between the original values and the random permutation
    """
    return lam * x + (1 - lam) * x[random_index, :]


def cutmix(x: torch.Tensor, random_index: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """Define how apply cutmix to a tensor

    :param x: Tensor on which apply the cutmix augmentation
    :param random_index: list of indices used to permute the tensor
    :param lam: probability values have 0 in a binary random mask, so it means probability original values will
    be updated
    """
    x_binary_mask = torch.from_numpy(np.random.choice(2, size=x.shape, p=[lam, 1 - lam]))
    x_random = x[random_index, :]
    x_noised = x.clone().detach()
    x_noised[x_binary_mask == 0] = x_random[x_binary_mask == 0]
    return x_noised


def get_random_index(x: torch.Tensor) -> torch.Tensor:
    """Given a tensor it compute random indices between 0 and the number of the first dimension

    :param x: Tensor used to get the number of rows
    """
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    return index
