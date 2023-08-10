import numpy as np


def jpp(zeta, psi, dx, i, j):
    return (
        (zeta[i + 1, j] - zeta[i - 1, j]) * (psi[i, j + 1] - psi[i, j - 1])
        - (zeta[i, j + 1] - zeta[i, j - 1]) * (psi[i + 1, j] - psi[i - 1, j])
    ) / (4 * dx**2)


def jpx(zeta, psi, dx, i, j):
    return (
        zeta[i + 1, j] * (psi[i + 1, j + 1] - psi[i + 1, j - 1])
        - zeta[i - 1, j] * (psi[i - 1, j + 1] - psi[i - 1, j - 1])
        - zeta[i, j + 1] * (psi[i + 1, j + 1] - psi[i - 1, j + 1])
        + zeta[i, j - 1] * (psi[i + 1, j - 1] - psi[i - 1, j - 1])
    ) / (4 * dx**2)


def jxp(zeta, psi, dx, i, j):
    return (
        zeta[i + 1, j + 1] * (psi[i, j + 1] - psi[i + 1, j])
        - zeta[i - 1, j - 1] * (psi[i - 1, j] - psi[i, j - 1])
        - zeta[i - 1, j + 1] * (psi[i, j + 1] - psi[i - 1, j])
        + zeta[i + 1, j - 1] * (psi[i + 1, j] - psi[i, j - 1])
    ) / (4 * dx**2)


def arakawa(zeta, psi, dx):
    val = np.empty_like(zeta)
    for i in range(1, zeta.shape[0] - 1):
        for j in range(1, zeta.shape[1] - 1):
            val[i][j] = (
                jpp(zeta, psi, dx, i, j)
                + jpx(zeta, psi, dx, i, j)
                + jxp(zeta, psi, dx, i, j)
            )
    return val / 3


def periodic_arakawa(zeta, psi, dx):
    return arakawa(np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx)[
        1:-1, 1:-1
    ]


## Vectorized


def arakawa_vec(zeta, psi, dx):
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


def periodic_arakawa_vec(zeta, psi, dx):
    return arakawa_vec(np.pad(zeta, 1, mode="wrap"), np.pad(psi, 1, mode="wrap"), dx)
