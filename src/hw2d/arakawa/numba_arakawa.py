import numpy as np
from numba import stencil, jit, prange


@stencil
def jpp_nb(zeta, psi, d):
    return ((zeta[1, 0] - zeta[-1, 0])*(psi[0, 1] - psi[0, -1])
            - (zeta[0, 1] - zeta[0, -1])*(psi[1, 0] - psi[-1, 0]))/(4*d**2)


@stencil
def jpx_nb(zeta, psi, d):
    return (zeta[1, 0]*(psi[1, 1] - psi[1, -1])
            - zeta[-1, 0]*(psi[-1, 1] - psi[-1, -1])
            - zeta[0, 1]*(psi[1, 1] - psi[-1, 1])
            + zeta[0, -1]*(psi[1, -1] - psi[-1, -1]))/(4*d**2)


@stencil
def jxp_nb(zeta, psi, d):
    return (zeta[1,  1]*(psi[0, 1] - psi[1,  0])
            - zeta[-1, -1]*(psi[-1, 0] - psi[0, -1])
            - zeta[-1,  1]*(psi[0, 1] - psi[-1,  0])
            + zeta[1, -1]*(psi[1, 0] - psi[0, -1]))/(4*d**2)


@jit(nopython=True, parallel=True, nogil=True)
def arakawa_nb(zeta, psi, d):
    return (jpp(zeta, psi, d) + jpx(zeta, psi, d) + jxp(zeta, psi, d))/3


def periodic_arakawa_nb(zeta, psi, d):
    return arakawa(np.pad(zeta, 1, mode='wrap'), np.pad(psi, 1, mode='wrap'), d)[1:-1, 1:-1]


# Full Stencil
@stencil
def arakawa_stencil(zeta, psi):
    return (zeta[1, 0] * (psi[0, 1] - psi[0, -1] + psi[1, 1] - psi[1, -1])
           -zeta[-1, 0] * (psi[0, 1] - psi[0, -1] + psi[-1, 1] - psi[-1, -1])
           -zeta[0, 1] * (psi[1, 0] - psi[-1, 0] + psi[1, 1] - psi[-1, 1])
           +zeta[0, -1] * (psi[1, 0] - psi[-1, 0] + psi[1, -1] - psi[-1, -1])
           +zeta[1, -1] * (psi[1, 0] - psi[0, -1])
           +zeta[1, 1] * (psi[0, 1] - psi[1, 0])
           -zeta[-1, 1] * (psi[0, 1] - psi[-1, 0])
           -zeta[-1, -1] * (psi[-1, 0] - psi[0, -1]))

@jit(nopython=True)
def arakawa_stencil_full(zeta, psi, d):
    return (arakawa_stencil(zeta, psi))[1:-1, 1:-1]/(12*d**2)

def periodic_arakawa_stencil(zeta, psi, d):
    return arakawa_stencil_full(np.pad(zeta, 1, mode='wrap'), np.pad(psi, 1, mode='wrap'), d)

## Vectorized

@jit(nopython=True)
def arakawa_vec(zeta, psi, d):
    """2D periodic first-order Arakawa
    requires 1 cell padded input on each border"""
    return   (zeta[2:, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[2:, 2:] - psi[2:, 0:-2])
            - zeta[0:-2, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[0:-2, 2:] - psi[0:-2, 0:-2])
            - zeta[1:-1, 2:] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 2:] - psi[0:-2, 2:])
            + zeta[1:-1, 0:-2] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 0:-2] - psi[0:-2, 0:-2])
            + zeta[2:, 0:-2] * (psi[2:, 1:-1] - psi[1:-1, 0:-2])
            + zeta[2:, 2:] * (psi[1:-1, 2:] - psi[2:, 1:-1])
            - zeta[0:-2, 2:] * (psi[1:-1, 2:] - psi[0:-2, 1:-1])
            - zeta[0:-2, 0:-2] * (psi[0:-2, 1:-1] - psi[1:-1, 0:-2])) / (12 * d**2)

def periodic_arakawa_vec(zeta, psi, d):
    return arakawa_vec(np.pad(zeta, 1, mode='wrap'), np.pad(psi, 1, mode='wrap'), d)
