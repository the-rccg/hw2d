import numpy as np
import pytest
from hw2d.initializations.fourier_noise import get_fft_noise
from hw2d.poisson_solvers.numpy_fourier_poisson import fourier_poisson_double, fourier_poisson_numpy
fourier_poisson_np = fourier_poisson_double
from hw2d.poisson_solvers.numba_fourier_poisson import fourier_poisson_double
fourier_poisson_nb = fourier_poisson_double
from hw2d.gradients.numpy_gradients import periodic_laplace


TEST_SIZE = 32
defs = dict(atol=1e-14, rtol=1e-20, equal_nan=False)
defs = dict(atol=1e-7, rtol=1e-20, equal_nan=False)


def test_fourier_poisson():
    a = np.random.rand(TEST_SIZE,TEST_SIZE)
    dx = float(np.random.rand(1))
    result = fourier_poisson_nb(a, dx)
    expected = fourier_poisson_np(a, dx)
    print(result)
    print(expected)
    # Assert that shapes match
    assert result.shape == expected.shape, f"Expected shape: {expected.shape}, but got {result.shape}"

    # Assert that values are close
    is_close = np.isclose(result, expected, **defs)

    # If not all values are close, print detailed info
    if not np.all(is_close):
        abs_diff = np.abs(result - expected)
        rel_diff = abs_diff / np.abs(expected)
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        
        msg = (f"Max absolute difference: {max_abs_diff}\n"
               f"Max relative difference: {max_rel_diff}\n")
        
        assert False, msg


def test_poisson():
    def get_2d_sine(grid_size, L):
        indices = np.array(np.meshgrid(*list(map(range, grid_size))))
        phys_coord = indices.T*L/(grid_size[0])  # between [0, L)
        x, y = phys_coord.T
        d = np.sin(x+1) * np.sin(y+1)
        return d

    N = 512
    L = 2*np.pi
    dx = L/N

    sin = get_2d_sine((N,N), L)
    laplace_sin = periodic_laplace(sin, dx)
    pred_sins = {
        "fourier_poisson_np": fourier_poisson_np(laplace_sin, dx),
        "fourier_poisson_nb": fourier_poisson_nb(laplace_sin, dx),
        "fourier_poisson_numpy": fourier_poisson_numpy(laplace_sin, dx)
    }

    for k, v in pred_sins.items():
        print(np.max(np.abs(v-sin)), np.mean(v-sin))

    fft_noise = get_fft_noise(
        (N, N),
        L,
        1,
        min_wavelength=dx * 12,
        max_wavelength=dx * N * 100,
        factor=2,
    )
    laplace_fft_noise = periodic_laplace(fft_noise, dx)
    pred_fft_noise = {
        "fourier_poisson_np": fourier_poisson_np(laplace_fft_noise, dx),
        "fourier_poisson_nb": fourier_poisson_nb(laplace_fft_noise, dx),
        "fourier_poisson_numpy":  fourier_poisson_numpy(laplace_fft_noise, dx)
    }

    for k, v in pred_fft_noise.items():
        print(np.max(np.abs(v-fft_noise)), np.mean(v-fft_noise))


if __name__ == '__main__':
    pytest.main()
