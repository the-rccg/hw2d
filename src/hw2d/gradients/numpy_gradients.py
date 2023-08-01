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
        padded[0:-2, 1:-1]  # above
        + padded[1:-1, 0:-2]  # left
        - 4 * padded[1:-1, 1:-1]  # center
        + padded[1:-1, 2:]  # right
        + padded[2:, 1:-1]  # below
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
    return laplace(np.pad(arr, 1, "wrap"), dx)


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
    padded = np.pad(input_field, 1, mode="wrap")
    return gradient(padded, dx, axis=axis)
