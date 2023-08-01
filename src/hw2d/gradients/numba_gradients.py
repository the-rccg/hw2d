import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, parallel=True)
def laplace(padded: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with Numba optimization.

    Args:
        padded (np.ndarray): 2D array with padding of size 1.
        dx (float): The spacing between grid points.

    Returns:
        np.ndarray: The gradient of the input array.
    """
    return (
        padded[0:-2, 1:-1]  # above
        + padded[1:-1, 0:-2]  # left
        - 4 * padded[1:-1, 1:-1]  # center
        + padded[1:-1, 2:]  # right
        + padded[2:, 1:-1]  # below
    ) / dx**2


def periodic_laplace(arr: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions and Numba optimization.

    Args:
        a (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.

    Returns:
        np.ndarray: The gradient of the input array with periodic boundary conditions.
    """
    return laplace(np.pad(arr, 1, "wrap"), dx)


@jit(nopython=True)
def laplace_N(arr: np.ndarray, dx: float, N: int) -> np.ndarray:
    """
    Compute the gradient of a 2D array N times successively.

    Args:
        arr (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        N (int): Number of iterations.

    Returns:
        np.ndarray: The gradient of the input array after N iterations.
    """
    for _ in range(N):
        arr = laplace(arr, dx)
    return arr


def periodic_laplace_N(arr: np.ndarray, dx: float, N: int) -> np.ndarray:
    """
    Compute the gradient of a 2D array N times successively with periodic boundary conditions.

    Args:
        arr (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        N (int): Number of iterations.

    Returns:
        np.ndarray: The gradient of the input array after N iterations with periodic boundary conditions.
    """
    return laplace_N(np.pad(arr, N, "wrap"), dx, N)


@jit(nopython=True)
def gradient(padded: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with Numba optimization.

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


def periodic_gradient(input_field: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary condition with Numba optimization

    Args:
        padded (np.ndarray): 2D array with padding of size 1.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        np.ndarray: Gradient in axis-direction with periodic boundary conditions.
    """
    padded = np.pad(input_field, 1, mode="wrap")
    return gradient(padded, dx, axis=axis)
