import numpy as np
import numba

precision = np.float64
complex_precision = np.complex128


def fourier_poisson_double(tensor: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
    """Inverse operation to `fourier_laplace`."""
    tensor = np.array(tensor, dtype=complex_precision)
    frequencies = np.fft.fft2(tensor)

    k = fftfreq_sq(tensor.shape)
    result_comp = core_computation(frequencies, k, dx, times)

    result = np.real(np.fft.ifft2(result_comp))
    return result.astype(precision)


@numba.jit(nopython=True)
def core_computation(
    frequencies: np.ndarray, k: np.ndarray, dx: float, times: int = 1
) -> np.ndarray:
    fft_laplace = -((2 * np.pi) ** 2) * k
    # Avoiding the use of np.inf for now. Set to a very large number.
    fft_laplace[0, 0] = 1e14

    frequencies = np.where(fft_laplace == 0, 0, frequencies)
    result = frequencies / (fft_laplace**times)

    return result * dx**2


@numba.jit(nopython=True)
def fftfreq_sq(resolution: np.ndarray) -> np.ndarray:
    dim_x, dim_y = resolution
    freq_x = custom_fftfreq(dim_x)
    freq_y = custom_fftfreq(dim_y)

    # Equivalent to:  k_sq = freq_x**2 + freq_y**2
    k_sq = np.empty((dim_x, dim_y), dtype=precision)
    for i in numba.prange(dim_x):
        for j in numba.prange(dim_y):
            k_sq[i, j] = freq_x[i] ** 2 + freq_y[j] ** 2

    return k_sq


@numba.jit(nopython=True)
def custom_fftfreq(n: np.ndarray) -> np.ndarray:
    """Custom FFT frequency function to replicate np.fft.fftfreq for Numba."""
    results = np.empty(n, dtype=np.int16)
    N = (n - 1) // 2 + 1
    results[:N] = np.arange(0, N)
    results[N:] = np.arange(-(n // 2), 0)
    return results / n
