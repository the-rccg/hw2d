import numpy as np
import pytest
from hw2d.gradients import numpy_gradients
from hw2d.gradients.numba_gradients import (
    laplace,
    periodic_laplace,
    laplace_N,
    periodic_laplace_N,
    gradient,
    periodic_gradient,
)


TEST_SIZE = 32


def test_laplace_nb():
    a = np.random.rand(TEST_SIZE, TEST_SIZE)
    dx = float(np.random.rand(1))
    result = laplace(np.pad(a, 1), dx)
    expected = numpy_gradients.laplace(np.pad(a, 1), dx)
    assert np.array_equal(result, expected)


def test_periodic_laplace():
    a = np.random.rand(TEST_SIZE, TEST_SIZE)
    dx = float(np.random.rand(1))
    result = periodic_laplace(a, dx)
    expected = numpy_gradients.periodic_laplace(a, dx)
    assert np.array_equal(result, expected)


def test_periodic_laplace_2():
    a = np.random.rand(TEST_SIZE, TEST_SIZE)
    dx = float(np.random.rand(1))
    result = periodic_laplace_N(a, N=2, dx=dx)
    expected = numpy_gradients.periodic_laplace_N(a, dx, N=2)
    assert np.array_equal(result, expected)


def test_periodic_laplace_3():
    a = np.random.rand(TEST_SIZE, TEST_SIZE)
    dx = float(np.random.rand(1))
    result = periodic_laplace_N(a, N=3, dx=dx)
    expected = numpy_gradients.periodic_laplace_N(a, dx, N=3)
    assert np.array_equal(result, expected)


def test_gradient():
    a = np.random.rand(TEST_SIZE, TEST_SIZE)
    dx = float(np.random.rand(1))
    result = gradient(a, dx, axis=0)
    expected = numpy_gradients.gradient(a, dx, axis=0)
    assert np.array_equal(result, expected)


def test_periodic_gradient():
    a = np.random.rand(TEST_SIZE, TEST_SIZE)
    dx = float(np.random.rand(1))
    result = periodic_gradient(a, dx, axis=0)
    expected = numpy_gradients.periodic_gradient(a, dx, axis=0)
    assert np.array_equal(result, expected)


if __name__ == "__main__":
    pytest.main()
