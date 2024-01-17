"""
NumPy-based Arakawa Scheme Implementations
==========================================

This module provides implementations of the Arakawa scheme for computing the Poisson bracket using the NumPy library. 
The functions are tailored for a 2D domain and utilize efficient NumPy array operations to achieve accurate and fast computations.

The module includes:

- Basic Jacobian functions with different discretization strategies (`jpp`, `jpx`, `jxp`).
- A main function (`arakawa`) which computes the Poisson bracket as an average of the Jacobians.
- Periodically padded implementations (`periodic_arakawa`).
- A vectorized implementation for improved performance (`arakawa_vec`).

Note: The functions in this module require the vorticity field (`zeta`) and the stream function field (`psi`) as primary inputs.
"""

import numpy as np


def jpp(zeta, psi, dx, i, j):
    """
    Compute the Jacobian using centered differences for both fields.

    Args:
        zeta (np.ndarray): Vorticity field.
        psi (np.ndarray): Stream function field.
        dx (float): Grid spacing.
        i, j (int): Indices for spatial location (x, y respectively)

    Returns:
        float: Jacobian value at (i, j).
    """
    return (
        (zeta[j, i + 1] - zeta[j,i - 1]) * (psi[j + 1, i] - psi[j - 1, i])
        - (zeta[j + 1, i] - zeta[j - 1, i]) * (psi[j, i + 1] - psi[j, i - 1])
    ) / (4 * dx**2)


def jpx(zeta, psi, dx, i, j):
    """
    Compute the Jacobian using centered differences for the vorticity field
    and staggered differences for the stream function field.

    Args:
        zeta (np.ndarray): Vorticity field.
        psi (np.ndarray): Stream function field.
        dx (float): Grid spacing.
        i, j (int): Indices for spatial location (x, y respectively)

    Returns:
        float: Jacobian value at (i, j).
    """
    return (
        zeta[j, i + 1] * (psi[j + 1, i + 1] - psi[j - 1, i + 1])
        - zeta[j, i - 1] * (psi[j + 1, i - 1] - psi[j - 1, i - 1])
        - zeta[j + 1, i] * (psi[j + 1, i + 1] - psi[j + 1, i - 1])
        + zeta[j - 1, i] * (psi[j - 1, i + 1] - psi[j - 1, i - 1])
    ) / (4 * dx**2)


def jxp(zeta, psi, dx, i, j):
    """
    Compute the Jacobian using staggered differences for the vorticity field
    and centered differences for the stream function field.

    Args:
        zeta (np.ndarray): Vorticity field.
        psi (np.ndarray): Stream function field.
        dx (float): Grid spacing.
        i, j (int): Indices for spatial location (x, y respectively)

    Returns:
        float: Jacobian value at (i, j).
    """
    return (
        zeta[j + 1, i + 1] * (psi[j + 1, i] - psi[j, i + 1])
        - zeta[j - 1, i - 1] * (psi[j, i - 1] - psi[j - 1, i])
        - zeta[j + 1, i - 1] * (psi[j + 1, i] - psi[j, i - 1])
        + zeta[j - 1, i + 1] * (psi[j, i + 1] - psi[j - 1, i])
    ) / (4 * dx**2)


def arakawa(zeta, psi, dx):
    """
    Compute the Poisson bracket as an average Jacobian using the Arakawa scheme.

    Args:
        zeta (np.ndarray): Vorticity field.
        psi (np.ndarray): Stream function field.
        dx (float): Grid spacing.

    Returns:
        np.ndarray: Compute the Poisson bracket as an average Jacobian over the entire grid.
    """
    val = np.empty_like(zeta)
    for j in range(1, zeta.shape[0] - 1):  # j = y
        for i in range(1, zeta.shape[1] - 1):  # i = x
            val[j][i] = (
                jpp(zeta, psi, dx, i, j)
                + jpx(zeta, psi, dx, i, j)
                + jxp(zeta, psi, dx, i, j)
            )
    return val / 3


def periodic_arakawa(zeta, psi, dx):
    """
    Compute the Arakawa Scheme with periodic boundary conditions.

    Args:
        zeta (np.ndarray): Vorticity field.
        psi (np.ndarray): Stream function field.
        dx (float): Grid spacing.

    Returns:
        np.ndarray: Compute the Poisson bracket as an average Jacobian over the grid without padding.
    """
    return arakawa(np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx)[
        1:-1, 1:-1
    ]


## Vectorized


def arakawa_vec(zeta, psi, dx):
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function
    using a vectorized version of the Arakawa scheme. This function is designed
    for a 2D periodic domain and requires a 1-cell padded input on each border.

    Args:
        zeta (np.ndarray): Vorticity field with padding.
        psi (np.ndarray): Stream function field with padding.
        dx (float): Grid spacing.

    Returns:
        np.ndarray: Discretized Poisson bracket (Jacobian) over the grid.
    """
    return (
        zeta[1:-1, 2:] * (psi[2:,1:-1] - psi[0:-2, 1:-1] + psi[2:, 2:] - psi[0:-2, 2:])
        - zeta[1:-1,0:-2]
        * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 0:-2] - psi[0:-2, 0:-2])
        - zeta[2:, 1:-1]
        * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[2:, 2:] - psi[2:, 0:-2])
        + zeta[0:-2, 1:-1]
        * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[0:-2, 2:] - psi[0:-2, 0:-2])
        + zeta[0:-2, 2:] * (psi[1:-1, 2:] - psi[0:-2, 1:-1])
        + zeta[2:, 2:] * (psi[2:, 1:-1] - psi[1:-1, 2:])
        - zeta[2:, 0:-2] * (psi[2:, 1:-1] - psi[1:-1, 0:-2])
        - zeta[0:-2, 0:-2] * (psi[1:-1, 0:-2] - psi[0:-2, 1:-1])
    ) / (12 * dx**2)


def periodic_arakawa_vec(zeta, psi, dx):
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function for a 2D periodic
    domain using a vectorized version of the Arakawa scheme. This function automatically
    handles the required padding.

    Args:
        zeta (np.ndarray): Vorticity field.
        psi (np.ndarray): Stream function field.
        dx (float): Grid spacing.

    Returns:
        np.ndarray: Discretized Poisson bracket (Jacobian) over the grid without padding.
    """
    return arakawa_vec(np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx)
