import numpy as np
import scipy.signal
import pytest
from unittest import TestCase
from hw2d.initializations.sine import get_2d_sine
from hw2d.gradients.numpy_gradients import (
    laplace,
    periodic_laplace,
    fourier_laplace,
    gradient,
    periodic_gradient,
)


class test_gradients(TestCase):
    def test_2d(self):
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

    def test_3d(self):
        N = 128
        time = 10
        grid_size = (N, N)
        L = 1
        dx = 0.2
        input_field = get_2d_sine(grid_size, L)
        arr = scipy.signal.convolve2d(
            input_field, [[0, 1, 0], [1, -4, 1], [0, 1, 0]], boundary="wrap"
        )[1:-1, 1:-1] / (dx**2)
        input_field = np.stack([input_field] * time, axis=0)
        arr = np.stack([arr] * time, axis=0)
        arr2 = periodic_laplace(input_field, dx)
        assert np.allclose(arr, arr2), f"{arr.shape}, {arr2.shape}"


if __name__ == "__main__":
    pytest.main()
