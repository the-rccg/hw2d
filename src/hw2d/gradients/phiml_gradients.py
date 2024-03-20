"""
PhiML-based Gradient Computation
================================

This module offers a collection of functions for computing gradients on 2D grids using the NumPy library.
It provides a standard implementation based on array computation suitable for solving the HW2D model, including:

- Basic Gradient Computation (`gradient`): Computes the gradient of a 2D array using central finite differences.
- Periodic Gradient (`periodic_gradient`): Computes the gradient with periodic boundary conditions.
- Laplace Operations:
    - Basic Laplace Computation (`laplace`): Computes the Laplace using finite differences.
    - Periodic Laplace (`periodic_laplace`): Laplace operation with periodic boundary conditions.
    - Iterative Laplace (`periodic_laplace_N`): Computes the Laplace N times successively.
    - Fourier-based Laplace (`fourier_laplace`): Computes the Laplace using Fourier transforms for enhanced accuracy.

All functions in this module are optimized for performance while ensuring accuracy, making them suitable for both prototyping and production-level simulations.
"""
from phiml import math as pm
from phiml.math.extrapolation import PERIODIC


def laplace(padded: pm.Tensor, dx: float):
    """
    Compute the Laplace of a 2D array using finite differences.

    Args:
        padded (pm.Tensor): 2D array with padding of size 1.
        dx (float): The spacing between grid points.

    Returns:
        pm.Tensor: The Laplace of the input array.
    """
    laplace_kernel = pm.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])

    return pm.convolve(padded, kernel=laplace_kernel, mode=PERIODIC) / dx**2


def periodic_laplace(arr: pm.Tensor, dx: float):
    """
    Compute the Laplace of a 2D array using finite differences with periodic boundary conditions.

    Args:
        a (pm.Tensor): Input 2D array.
        dx (float): The spacing between grid points.

    Returns:
        pm.Tensor: The Laplace of the input array with periodic boundary conditions.
    """
    laplace_kernel = pm.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], pm.spatial("x", "y"))

    return pm.convolve(arr, kernel=laplace_kernel, extrapolation=PERIODIC) / dx**2


def periodic_laplace_N(arr: pm.Tensor, dx: float, N: int = 1) -> pm.Tensor:
    """
    Compute the Laplace of a 2D array using finite differences N times successively with periodic boundary conditions.

    Args:
        a (pm.Tensor): Input 2D array.
        dx (float): The spacing between grid points.
        N (int): Number of iterations.

    Returns:
        pm.Tensor: The Laplace of the input array with periodic boundary conditions.
    """
    for _ in range(N):
        arr = periodic_laplace(arr, dx)
    return arr


def fourier_laplace(grid: pm.Tensor, dx: float, N: int = 1) -> pm.Tensor:
    """
    Compute the Laplace of a 2D array using Fourier transform.

    Args:
        grid (pm.Tensor): Input 2D array.
        dx (float): The spacing between grid points.
        times (int, optional): Number of times to apply the Laplace operator. Default is 1.

    Returns:
        pm.Tensor: The Laplace of the input array using Fourier transform.
    """
    frequencies = pm.fft(grid)
    k_squared = pm.sum(pm.fftfreq(grid.shape) ** 2, dim=pm.channel)
    fft_laplace = -((2 * pm.pi) ** 2) * k_squared
    result = pm.real(pm.ifft(frequencies * (fft_laplace**N)))
    return (result / dx**2)


def gradient(padded: pm.Tensor, dx: float, axis: int = 0) -> pm.Tensor:
    """
    Compute the gradient of a 2D array using finite differences.

    Args:
        padded (pm.Tensor): 2D array with padding of size 1.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        pm.Tensor: Gradient in axis-direction.
    """
    if axis == 0:
        return (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / (2 * dx)
    elif axis == 1:
        return (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / (2 * dx)
    elif axis == -2:
        return (padded[..., 2:, 1:-1] - padded[..., 0:-2, 1:-1]) / (2 * dx)
    elif axis == -1:
        return (padded[..., 1:-1, 2:] - padded[..., 1:-1, 0:-2]) / (2 * dx)


def periodic_gradient(input_field: pm.Tensor, dx: float, axis: str) -> pm.Tensor:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions.

    Args:
        input_field (pm.Tensor): Input 2D array.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        tuple: Gradient in y-direction, gradient in x-direction with periodic boundary conditions.
    """
    return pm.spatial_gradient(input_field, dx=dx, dims=axis, difference="central", padding=PERIODIC, stack_dim=None)
