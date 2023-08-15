"""
Fourier-based Poisson Solvers with NumPy
========================================

This module provides a set of functions to solve the Poisson equation using the Fourier transform approach and the NumPy library. The Fourier method is especially beneficial for periodic domains and spectral methods, as it can retrieve the original field from its gradient or Laplace efficiently in the spectral space.

Main functionalities include:

- `fourier_poisson_double`: Solves the Poisson equation using double precision (complex128).
- `fourier_poisson_single`: Solves the Poisson equation using single precision (complex64).
- `fourier_poisson_numpy`: A more general Poisson solver that auto-detects the input type.

These functions are designed for both prototyping and production-level simulations, offering a balance between accuracy and performance. They are particularly well-suited for large-scale simulations in periodic domains.
"""

import numpy as np


def fourier_poisson_double(tensor: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
    """Inverse operation to `fourier_laplace`."""
    tensor = np.array(tensor, dtype=np.complex128)
    frequencies = np.fft.fft2(tensor)

    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in tensor.shape], indexing="ij")
    k = np.stack(k, -1)
    k = np.sum(k**2, axis=-1)
    fft_laplace = -((2 * np.pi) ** 2) * k
    fft_laplace[0, 0] = np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        result = frequencies / (fft_laplace**times)
    result = np.where(fft_laplace == 0, 0, result)

    result = np.real(np.fft.ifft2(result))
    return (result * dx**2).astype(np.float64)


def fourier_poisson_single(tensor: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
    """Inverse operation to `fourier_laplace`."""
    tensor = np.array(tensor, dtype=np.complex64)
    frequencies = np.fft.fft2(tensor)

    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in tensor.shape], indexing="ij")
    k = np.stack(k, -1)
    k = np.sum(k**2, axis=-1)
    fft_laplace = -((2 * np.pi) ** 2) * k
    fft_laplace[0, 0] = np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        result = frequencies / (fft_laplace**times)
    result = np.where(fft_laplace == 0, 0, result)

    result = np.real(np.fft.ifft2(result))
    return (result * dx**2).astype(np.float32)


def fourier_poisson_numpy(grid: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
    """
    Inverse operation to `fourier_laplace`.

    Args:
      grid: numpy array
      dx: float or list or tuple
      times: int (Default value = 1)

    Returns:
      result: numpy array
    """
    # Convert grid to complex type
    grid_complex = np.asarray(grid, dtype=np.complex128)

    # Compute the FFT of the grid
    frequencies = np.fft.fftn(grid_complex)

    # Compute squared frequency magnitude
    k_squared = np.sum(
        np.array(np.meshgrid(*[np.fft.fftfreq(dim) for dim in grid.shape])) ** 2, axis=0
    )

    # Compute inverse of squared frequency multiplied by -(2 * np.pi)^2
    fft_laplace = -((2 * np.pi) ** 2) * k_squared

    # Handle division by zero safely and compute the inverse FFT
    divisor = fft_laplace**times
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_division = np.where(divisor != 0, frequencies / divisor, 0)
    result = np.real(np.fft.ifftn(safe_division))

    # Multiply by dx squared
    result *= np.prod(dx) ** 2

    # Cast result to the original datatype of grid
    return result.astype(grid.dtype)
