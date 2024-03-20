"""
PhiML-based Arakawa Scheme Implementations
==========================================

This module provides implementations of the Arakawa scheme for computing the Poisson bracket using the PhiML library.
The functions are tailored for a 2D domain and utilize efficient PhiML array operations to achieve accurate and fast computations.

The module includes:

- Basic Jacobian functions with different discretization strategies (`jpp`, `jpx`, `jxp`).
- A main function (`arakawa`) which computes the Poisson bracket as an average of the Jacobians.
- Periodically padded implementations (`periodic_arakawa`).
- A vectorized implementation for improved performance (`arakawa_vec`).

Note: The functions in this module require the vorticity field (`zeta`) and the stream function field (`psi`) as primary inputs.
"""
from phiml import math as pm
from phiml.math.extrapolation import PERIODIC


def jpp(zeta, psi, dx, i, j):
    """
    Compute the Jacobian using centered differences for both fields.

    Args:
        zeta (pm.Tensor): Vorticity field.
        psi (pm.Tensor): Stream function field.
        dx (float): Grid spacing.
        i, j (int): Indices for spatial location.

    Returns:
        float: Jacobian value at (i, j).
    """
    return (
        (zeta[i + 1, j] - zeta[i - 1, j]) * (psi[i, j + 1] - psi[i, j - 1])
        - (zeta[i, j + 1] - zeta[i, j - 1]) * (psi[i + 1, j] - psi[i - 1, j])
    ) / (4 * dx**2)


def jpx(zeta, psi, dx, i, j):
    """
    Compute the Jacobian using centered differences for the vorticity field
    and staggered differences for the stream function field.

    Args:
        zeta (pm.Tensor): Vorticity field.
        psi (pm.Tensor): Stream function field.
        dx (float): Grid spacing.
        i, j (int): Indices for spatial location.

    Returns:
        float: Jacobian value at (i, j).
    """
    return (
        zeta[i + 1, j] * (psi[i + 1, j + 1] - psi[i + 1, j - 1])
        - zeta[i - 1, j] * (psi[i - 1, j + 1] - psi[i - 1, j - 1])
        - zeta[i, j + 1] * (psi[i + 1, j + 1] - psi[i - 1, j + 1])
        + zeta[i, j - 1] * (psi[i + 1, j - 1] - psi[i - 1, j - 1])
    ) / (4 * dx**2)


def jxp(zeta, psi, dx, i, j):
    """
    Compute the Jacobian using staggered differences for the vorticity field
    and centered differences for the stream function field.

    Args:
        zeta (pm.Tensor): Vorticity field.
        psi (pm.Tensor): Stream function field.
        dx (float): Grid spacing.
        i, j (int): Indices for spatial location.

    Returns:
        float: Jacobian value at (i, j).
    """
    return (
        zeta[i + 1, j + 1] * (psi[i, j + 1] - psi[i + 1, j])
        - zeta[i - 1, j - 1] * (psi[i - 1, j] - psi[i, j - 1])
        - zeta[i - 1, j + 1] * (psi[i, j + 1] - psi[i - 1, j])
        + zeta[i + 1, j - 1] * (psi[i + 1, j] - psi[i, j - 1])
    ) / (4 * dx**2)


## Vectorized

@pm.jit_compile
def arakawa_vec(zeta: pm.Tensor, psi: pm.Tensor, dx: float or pm.Tensor) -> pm.Tensor:
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function
    using a vectorized version of the Arakawa scheme. This function is designed
    for a 2D periodic domain and requires a 1-cell padded input on each border.

    Args:
        zeta (pm.Tensor): Vorticity field with padding.
        psi (pm.Tensor): Stream function field with padding.
        dx (float): Grid spacing.

    Returns:
        pm.Tensor: Discretized Poisson bracket (Jacobian) over the grid.
    """
    return (
        zeta.y[2:].x[1:-1] * (psi.y[1:-1].x[2:] - psi.y[1:-1].x[0:-2] + psi.y[2:].x[2:] - psi.y[2:].x[0:-2])
        - zeta.y[0:-2].x[1:-1]
        * (psi.y[1:-1].x[2:] - psi.y[1:-1].x[0:-2] + psi.y[0:-2].x[2:] - psi.y[0:-2].x[0:-2])
        - zeta.y[1:-1].x[2:]
        * (psi.y[2:].x[1:-1] - psi.y[0:-2].x[1:-1] + psi.y[2:].x[2:] - psi.y[0:-2].x[2:])
        + zeta.y[1:-1].x[0:-2]
        * (psi.y[2:].x[1:-1] - psi.y[0:-2].x[1:-1] + psi.y[2:].x[0:-2] - psi.y[0:-2].x[0:-2])
        + zeta.y[2:].x[0:-2] * (psi.y[2:].x[1:-1] - psi.y[1:-1].x[0:-2])
        + zeta.y[2:].x[2:] * (psi.y[1:-1].x[2:] - psi.y[2:].x[1:-1])
        - zeta.y[0:-2].x[2:] * (psi.y[1:-1].x[2:] - psi.y[0:-2].x[1:-1])
        - zeta.y[0:-2].x[0:-2] * (psi.y[0:-2].x[1:-1] - psi.y[1:-1].x[0:-2])
    ) / (12 * dx**2)


def periodic_arakawa_vec(zeta: pm.Tensor, psi: pm.Tensor, dx: float or pm.Tensor) -> pm.Tensor:
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function for a 2D periodic
    domain using a vectorized version of the Arakawa scheme. This function automatically
    handles the required padding.

    Args:
        zeta (pm.Tensor): Vorticity field.
        psi (pm.Tensor): Stream function field.
        dx (float): Grid spacing.

    Returns:
        pm.Tensor: Discretized Poisson bracket (Jacobian) over the grid without padding.
    """
    zeta = pm.pad(zeta, widths={"y":(1,1),"x":(1,1)}, mode=PERIODIC)
    psi = pm.pad(psi, widths={"y":(1,1),"x":(1,1)}, mode=PERIODIC)
    result = arakawa_vec(zeta, psi, dx)
    return result

