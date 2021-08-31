import torch
import numpy as np


def mixup_data(x1, x2, lam=1.0, y=None):
    """Returns mixed inputs, pairs of targets"""
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b

    return mixed_x1, mixed_x2


def add_noise(x_categ, x_cont, noise_lambda=0.1):
    lam = noise_lambda
    batch_size = x_categ.size()[0]
    index = torch.randperm(batch_size)
    cat_corr = torch.from_numpy(np.random.choice(2, x_categ.shape, p=[lam, 1 - lam]))
    con_corr = torch.from_numpy(np.random.choice(2, x_cont.shape, p=[lam, 1 - lam]))
    x1, x2 = x_categ[index, :], x_cont[index, :]
    x_categ_corr, x_cont_corr = x_categ.clone().detach(), x_cont.clone().detach()
    x_categ_corr[cat_corr == 0] = x1[cat_corr == 0]
    x_cont_corr[con_corr == 0] = x2[con_corr == 0]
    return x_categ_corr, x_cont_corr