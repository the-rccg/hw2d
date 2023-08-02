from unittest import TestCase

import numpy as np
import pytest
import scipy.signal
from hw2d.gradients.gradients import (
    fourier_laplace,
    gradient,
    laplace,
    periodic_gradient,
    periodic_laplace,
)
from hw2d.initializations.sine import get_2d_sine


class test_gradients(TestCase):
    def test(self):
        N = 128
        grid_size = (N, N)
        L = 1
        dx = 0.2
        input_field = get_2d_sine(grid_size, L)
        arr = scipy.signal.convolve2d(
            input_field, [[0, 1, 0], [1, -4, 1], [0, 1, 0]], boundary="wrap"
        )[1:-1, 1:-1] / (dx**2)
        arr2 = periodic_laplace(input_field, dx)
        assert np.allclose(arr, arr2), f"{arr.shape}, {arr2.shape}"


if __name__ == "__main__":
    pytest.main()
