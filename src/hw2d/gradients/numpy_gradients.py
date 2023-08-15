"""
NumPy-based Gradient Computation
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

import numpy as np


def laplace(padded: np.ndarray, dx: float):
    """
    Compute the Laplace of a 2D array using finite differences.

    Args:
        padded (np.ndarray): 2D array with padding of size 1.
        dx (float): The spacing between grid points.

    Returns:
        np.ndarray: The Laplace of the input array.
    """
    return (
        padded[..., 0:-2, 1:-1]  # above
        + padded[..., 1:-1, 0:-2]  # left
        - 4 * padded[..., 1:-1, 1:-1]  # center
        + padded[..., 1:-1, 2:]  # right
        + padded[..., 2:, 1:-1]  # below
    ) / dx**2


def periodic_laplace(arr: np.ndarray, dx: float):
    """
    Compute the Laplace of a 2D array using finite differences with periodic boundary conditions.

    Args:
        a (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.

    Returns:
        np.ndarray: The Laplace of the input array with periodic boundary conditions.
    """
    pad_size = 1
    if len(arr.shape) > 2:
        pad_size = [(0, 0) for _ in range(len(arr.shape))]
        pad_size[-1] = (1, 1)
        pad_size[-2] = (1, 1)
    return laplace(np.pad(arr, pad_size, "wrap"), dx)


def periodic_laplace_N(arr: np.ndarray, dx: float, N: int) -> np.ndarray:
    """
    Compute the Laplace of a 2D array using finite differences N times successively with periodic boundary conditions.

    Args:
        a (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        N (int): Number of iterations.

    Returns:
        np.ndarray: The Laplace of the input array with periodic boundary conditions.
    """
    for _ in range(N):
        arr = periodic_laplace(arr, dx)
    return arr


def fourier_laplace(grid: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
    """
    Compute the Laplace of a 2D array using Fourier transform.

    Args:
        grid (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        times (int, optional): Number of times to apply the Laplace operator. Default is 1.

    Returns:
        np.ndarray: The Laplace of the input array using Fourier transform.
    """
    frequencies = np.fft.fft2(grid.astype(np.complex128))
    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in grid.shape], indexing="ij")
    k = np.expand_dims(np.stack(k, -1), 0)
    k = k.astype(np.float64)
    k_squared = np.sum(k**2, axis=-1)
    fft_laplace = -((2 * np.pi) ** 2) * k_squared
    result = np.real(np.fft.ifft2(frequencies * fft_laplace**times))
    return (result / dx**2).astype(grid.dtype)[0]


def gradient(padded: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences.

    Args:
        padded (np.ndarray): 2D array with padding of size 1.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        np.ndarray: Gradient in axis-direction.
    """
    if axis == 0:
        return (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / (2 * dx)
    elif axis == 1:
        return (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / (2 * dx)
    elif axis == -2:
        return (padded[..., 2:, 1:-1] - padded[..., 0:-2, 1:-1]) / (2 * dx)
    elif axis == -1:
        return (padded[..., 1:-1, 2:] - padded[..., 1:-1, 0:-2]) / (2 * dx)


def periodic_gradient(input_field: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions.

    Args:
        input_field (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        tuple: Gradient in y-direction, gradient in x-direction with periodic boundary conditions.
    """
    if axis < 0:
        pad_size = [(0, 0) for _ in range(len(input_field.shape))]
        pad_size[-1] = (1, 1)
        pad_size[-2] = (1, 1)
    else:
        pad_size = 1
    padded = np.pad(input_field, pad_width=pad_size, mode="wrap")
    return gradient(padded, dx, axis=axis)
