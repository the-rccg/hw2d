import numpy as np
from numba import stencil, jit, prange


@stencil
def jpp_nb(zeta: np.ndarray, psi: np.ndarray, d: float) -> np.ndarray:
    return (
        (zeta[1, 0] - zeta[-1, 0]) * (psi[0, 1] - psi[0, -1])
        - (zeta[0, 1] - zeta[0, -1]) * (psi[1, 0] - psi[-1, 0])
    ) / (4 * d**2)


@stencil
def jpx_nb(zeta: np.ndarray, psi: np.ndarray, d: float) -> np.ndarray:
    return (
        zeta[1, 0] * (psi[1, 1] - psi[1, -1])
        - zeta[-1, 0] * (psi[-1, 1] - psi[-1, -1])
        - zeta[0, 1] * (psi[1, 1] - psi[-1, 1])
        + zeta[0, -1] * (psi[1, -1] - psi[-1, -1])
    ) / (4 * d**2)


@stencil
def jxp_nb(zeta: np.ndarray, psi: np.ndarray, d: float) -> np.ndarray:
    return (
        zeta[1, 1] * (psi[0, 1] - psi[1, 0])
        - zeta[-1, -1] * (psi[-1, 0] - psi[0, -1])
        - zeta[-1, 1] * (psi[0, 1] - psi[-1, 0])
        + zeta[1, -1] * (psi[1, 0] - psi[0, -1])
    ) / (4 * d**2)


@jit(nopython=True, parallel=True, nogil=True)
def arakawa_nb(zeta: np.ndarray, psi: np.ndarray, dx: float) -> np.ndarray:
    return (jpp(zeta, psi, dx) + jpx(zeta, psi, dx) + jxp(zeta, psi, dx)) / 3


def periodic_arakawa_nb(zeta, psi, dx):
    return arakawa(np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx)[
        1:-1, 1:-1
    ]


# Full Stencil
@stencil
def arakawa_stencil(zeta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return (
        zeta[1, 0] * (psi[0, 1] - psi[0, -1] + psi[1, 1] - psi[1, -1])
        - zeta[-1, 0] * (psi[0, 1] - psi[0, -1] + psi[-1, 1] - psi[-1, -1])
        - zeta[0, 1] * (psi[1, 0] - psi[-1, 0] + psi[1, 1] - psi[-1, 1])
        + zeta[0, -1] * (psi[1, 0] - psi[-1, 0] + psi[1, -1] - psi[-1, -1])
        + zeta[1, -1] * (psi[1, 0] - psi[0, -1])
        + zeta[1, 1] * (psi[0, 1] - psi[1, 0])
        - zeta[-1, 1] * (psi[0, 1] - psi[-1, 0])
        - zeta[-1, -1] * (psi[-1, 0] - psi[0, -1])
    )


@jit(nopython=True)
def arakawa_stencil_full(zeta: np.ndarray, psi: np.ndarray, dx: float) -> np.ndarray:
    return (arakawa_stencil(zeta, psi))[1:-1, 1:-1] / (12 * dx**2)


def periodic_arakawa_stencil(
    zeta: np.ndarray, psi: np.ndarray, dx: float
) -> np.ndarray:
    return arakawa_stencil_full(
        np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx
    )


## Vectorized


@jit(nopython=True)
def arakawa_vec(zeta: np.ndarray, psi: np.ndarray, dx: float) -> np.ndarray:
    """2D periodic first-order Arakawa
    requires 1 cell padded input on each border"""
    return (
        zeta[2:, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[2:, 2:] - psi[2:, 0:-2])
        - zeta[0:-2, 1:-1]
        * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[0:-2, 2:] - psi[0:-2, 0:-2])
        - zeta[1:-1, 2:]
        * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 2:] - psi[0:-2, 2:])
        + zeta[1:-1, 0:-2]
        * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 0:-2] - psi[0:-2, 0:-2])
        + zeta[2:, 0:-2] * (psi[2:, 1:-1] - psi[1:-1, 0:-2])
        + zeta[2:, 2:] * (psi[1:-1, 2:] - psi[2:, 1:-1])
        - zeta[0:-2, 2:] * (psi[1:-1, 2:] - psi[0:-2, 1:-1])
        - zeta[0:-2, 0:-2] * (psi[0:-2, 1:-1] - psi[1:-1, 0:-2])
    ) / (12 * dx**2)


def periodic_arakawa_vec(zeta: np.ndarray, psi: np.ndarray, dx: float) -> np.ndarray:
    return arakawa_vec(np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx)
