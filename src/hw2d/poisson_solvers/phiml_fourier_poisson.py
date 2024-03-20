"""
Fourier-based Poisson Solvers with PhiML
========================================

This module provides a set of functions to solve the Poisson equation using the Fourier transform approach and the NumPy library. The Fourier method is especially beneficial for periodic domains and spectral methods, as it can retrieve the original field from its gradient or Laplace efficiently in the spectral space.

Main functionalities include:

- `fourier_poisson_double`: Solves the Poisson equation using double precision (complex128).
- `fourier_poisson_single`: Solves the Poisson equation using single precision (complex64).
- `fourier_poisson_numpy`: A more general Poisson solver that auto-detects the input type.

These functions are designed for both prototyping and production-level simulations, offering a balance between accuracy and performance. They are particularly well-suited for large-scale simulations in periodic domains.
"""

import phiml.math as pm


def fourier_poisson_single(tensor: pm.Tensor, dx: float, times: int = 1) -> pm.Tensor:
    """Inverse operation to `fourier_laplace`."""
    frequencies = pm.fft(tensor, dims=("y", "x"))
    k_squared = pm.sum(pm.fftfreq(tensor.shape)**2, dim="vector")
    fft_laplace = -((2 * pm.PI) ** 2) * k_squared
    divisor = fft_laplace**times
    #safe_division = pm.where(divisor != 0, frequencies / divisor, 0)
    safe_division = pm.safe_div(frequencies, divisor)
    result = pm.real(pm.ifft(safe_division))
    return (result * dx**2)

