from lit_saint.augmentations import get_random_index
import numpy as np
import torch


def test_get_random_index():
    torch.manual_seed(0)
    x = torch.Tensor([1, 2, 3])
    expected_result = torch.from_numpy(np.array([2, 0, 1]))
    result = get_random_index(x)
    torch.testing.assert_equal(result, expected_result)
