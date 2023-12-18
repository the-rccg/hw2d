"""
Numerical Properties using NumPy (`numpy_properties`)
=====================================================

This module provides a collection of functions to compute various properties and metrics related to a 2D Hasegawa-Wakatani system. 
It leverages the NumPy library for efficient computations on grid-based data. 
The provided functionalities help in understanding the physical and spectral properties of the system.

Specifically, the module includes:

- **Sources and Sinks** such as $\\Gamma_n$ and $\\Gamma_c$.
- **Energies** including total, kinetic, and potential energy.
- **Enstrophy** to quantify the system's vorticity content.
- **Dissipation Metrics** to understand the system's energy dissipation rate over time.
- **Spectral Properties** of various metrics for further analysis and verification.

Refer to each function's docstring for detailed information on their use and mathematical formulation.
"""
# Define Frame Properties
# Assume (..., y, x) as shape
import numpy as np
from typing import Tuple
from hw2d.gradients.numpy_gradients import periodic_laplace_N, periodic_gradient


# Gammas


def get_gamma_n(n: np.ndarray, p: np.ndarray, dx: float, dy_p=None) -> float:
    """
    Compute the average particle flux $(\\Gamma_n)$ using the formula:
    $$
        \\Gamma_n = - \\int{\\mathrm{d^2} x \; \\tilde{n} \\frac{\partial \\tilde{\\phi}}{\\partial y}}
    $$

    Args:
        n (np.ndarray): Density (or similar field).
        p (np.ndarray): Potential (or similar field).
        dx (float): Grid spacing.
        dy_p (np.ndarray, optional): Gradient of potential in the y-direction.
            Computed from `p` if not provided.

    Returns:
        float: Computed average particle flux value.
    """
    if dy_p is None:
        dy_p = periodic_gradient(p, dx=dx, axis=-2)  # gradient in y
    gamma_n = -np.mean((n * dy_p), axis=(-1, -2))  # mean over y & x
    return gamma_n


def get_gamma_c(n: np.ndarray, p: np.ndarray, c1: float, dx: float) -> float:
    """
    Compute the sink $\\Gamma_c$ using the formula:
    $$
        \\Gamma_c = c_1 \\int{\\mathrm{d^2} x \; (\\tilde{n} - \\tilde{\\phi})^2}
    $$

    Args:
        n (np.ndarray): Density (or similar field).
        p (np.ndarray): Potential (or similar field).
        c1 (float): Proportionality constant.
        dx (float): Grid spacing.

    Returns:
        float: Computed particle flux value.
    """
    gamma_c = c1 * np.mean((n - p) ** 2, axis=(-1, -2))  # mean over y & x
    return gamma_c


# Spectral Gamma_n


def get_gamma_n_ky(
    n: np.ndarray, p: np.ndarray, dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the spectral components $\\Gamma_n(k_y)$"""
    ky_max = n.shape[-2] // 2
    n_dft = np.fft.fft2(n, norm="ortho", axes=(-2, -1))  # n(ky, kx)
    p_dft = np.fft.fft2(p, norm="ortho", axes=(-2, -1))  # p(ky, kx)
    k_kx, k_ky = np.meshgrid(
        *[np.fft.fftfreq(int(i), d=dx) * 2 * np.pi for i in n.shape[-2:]]
    )  # k(ky, kx)
    gamma_n_k = n_dft * 1j * k_ky * np.conjugate(p_dft)  # gamma_n(ky, kx)
    integrated_gamma_n_k = np.mean(np.real(gamma_n_k), axis=-1)[
        ..., :ky_max
    ]  # gamma_n(ky)
    ky = k_ky[:ky_max, 0]
    return ky, integrated_gamma_n_k


def get_gamma_n_spectrally(n: np.ndarray, p: np.ndarray, dx: float) -> float:
    """Calculate the $\\Gamma_n = \\int{\\mathrm{d} k_y \; \\Gamma_n(k_y)}$"""
    ky, gamma_n_ky = get_gamma_n_ky(n=n, p=p, dx=dx)
    gamma_n = np.mean(gamma_n_ky, axis=-1)  # mean over ky
    return gamma_n


# Energy


def get_energy(n: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """Energy of the HW2D system, sum of thermal and kinetic energy
    $$
        E = \\frac{1}{2} \\int{
            \\mathrm {d^2} x \;
            \\left(n^2 + | \\nabla \\phi |^2 \\right)
        }
    $$
    """
    # Squared L2-Norm
    squared_norm_grad_phi = periodic_gradient(phi, dx=dx, axis=-1)**2 + periodic_gradient(
        phi, dx=dx, axis=-2
    )**2
    # Integrate, then divide by 2
    integral = np.mean((n**2) + squared_norm_grad_phi, axis=(-1, -2))
    return integral / 2


# Enstrophy


def get_enstrophy(n: np.ndarray, omega: np.ndarray, dx: float) -> np.ndarray:
    """Enstrophy of the HW2D system
    $$
        \\mathbf{U = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n - \\Omega)^2}}
                   = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n^2 - \\nabla^2 \\phi)^2}
    $$
    """
    # omega = omega - np.mean(omega, axis=(-1, -2), keepdims=True)
    integral = np.mean(((n - omega) ** 2), axis=(-1, -2))
    return integral / 2


def get_enstrophy_phi(n: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """Enstrophy of the HW2D system from phi
    $$
        \\mathbf{U = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n^2 - \\nabla^2 \\phi)^2}}
                   = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n - \\Omega)^2}
    $$
    """
    omega = periodic_laplace_N(phi, dx, N=1)
    omega -= np.mean(omega, axis=(-1, -2), keepdims=True)
    integral = np.mean(((n - omega) ** 2), axis=(-1, -2))
    return integral / 2


def get_D(arr: np.ndarray, nu: float, N: int, dx: float) -> np.ndarray:
    """Calculate the hyperdiffusion coefficient
    $$
        \\nu \; \\nabla F
    $$

    Args:
        arr (np.ndarray): Field to work on
        nu (float): hyperdiffusion coefficient
        N (int): order of hyperdiffusion
        dx (float): grid spacing

    Returns:
        np.ndarray: _description_
    """
    return nu * periodic_laplace_N(arr, dx=dx, N=N)


# Sinks


def get_DE(n: np.ndarray, p: np.ndarray, Dn: np.ndarray, Dp: np.ndarray) -> float:
    """
    $$
        DE = \\int{\\mathrm{d^2} x \; n \; D_n - \\phi \; D_p}
    $$

    Args:
        n (np.ndarray): density field $n$
        p (np.ndarray): potential field $\\phi$
        Dn (np.ndarray): hyperdiffusion of the density field $n$
        Dp (np.ndarray): hyperdiffusion of the potential field $\\phi$

    Returns:
        float: Value of DE
    """
    DE = np.mean(n * Dn - p * Dp, axis=(-1, -2))
    return DE


def get_DU(n: np.ndarray, o: np.ndarray, Dn: np.ndarray, Dp: np.ndarray) -> float:
    """
    $$
        DU = \\int{\\mathrm{d^2} x \; (n - \Omega)  (D_n - D_\\phi)}
    $$

    Args:
        n (np.ndarray): density field $n$
        o (np.ndarray): potential field $\\phi$
        Dn (np.ndarray): hyperdiffusion of the density field $n$
        Dp (np.ndarray): hyperdiffusion of the potential field $\\phi$

    Returns:
        float: Value of D()
    """
    DE = -np.mean((n - o) * (Dn - Dp), axis=(-1, -2))
    return DE


# Time Variation


def get_dE_dt(gamma_n: np.ndarray, gamma_c: np.ndarray, DE: np.ndarray) -> float:
    """
    $$
        \\partial_t E = \\Gamma_n - \\Gamma_c - DE
    $$

    Args:
        gamma_n (np.ndarray): $\\Gamma_n$ 
        gamma_c (np.ndarray): $\\Gamma_c$
        DE (np.ndarray): $DE = n\; D_n - \\phi \; D_p$

    Returns:
        float: Time gradient of energy (in-/outflow)
    """
    return gamma_n - gamma_c - DE


def get_dU_dt(gamma_n: np.ndarray, DU: np.ndarray) -> float:
    """
    $$
        \\partial_t U = \\Gamma_n - DU
    $$

    Args:
        gamma_n (np.ndarray): $\\Gamma_n$
        DU (np.ndarray): $DU$

    Returns:
        float:  Time gradient of enstrophy (in-/outflow)
    """
    return gamma_n - DU


# Spectral Energies


def get_energy_N_ky(n: np.ndarray) -> np.ndarray:
    """thermal energy spectrum
    $$
        E^N(k_y) = \\frac{1}{2} |n(k_y)|^2
    $$
    """
    n_dft = np.fft.fft2(n, norm="ortho")
    # n_dft = np.mean(n_dft, axis=-1)
    E_N_ky = np.abs(n_dft) ** 2 / 2
    E_N_ky = np.mean(E_N_ky, axis=-1)
    return E_N_ky


def get_energy_N_spectrally(n: np.ndarray) -> np.ndarray:
    """thermal energy spectrally integrated
    $$
        E^N = \\int{\\mathrm{d} k_y \; E^N (k_y)} = \\int{\\mathrm{d} k_y \; \\frac{1}{2} |n(k)|^2}
    $$
    """
    E_N_ky = get_energy_N_ky(n)
    E_N = np.mean(E_N_ky, axis=-1)
    return E_N


def get_energy_V_ky(p: np.ndarray, dx: float) -> np.ndarray:
    """kinetic energy spectrum
    $$
        E^V(k_y) = \\frac{1}{2} | k_y \\phi(k_y) |^2
    $$
    """
    k_kx, k_ky = np.meshgrid(
        *[np.fft.fftfreq(int(i), d=dx) * 2 * np.pi for i in p.shape[-2:]]
    )  # k(ky, kx)
    p_dft = np.fft.fft2(p, norm="ortho")
    squared_norm_k = k_ky**2 + k_kx**2
    E_V_ky = np.abs(squared_norm_k * p_dft * np.conjugate(p_dft)) / 2
    E_V_ky = np.mean(E_V_ky, axis=-1)
    return E_V_ky


def get_energy_V_spectrally(p: np.ndarray, dx: float) -> np.ndarray:
    """kinetic energy spectrally integrated
    $$
        E^V = \\int{\\mathrm{d} k_y \; E^V(k_y)} 
            = \\int{\\mathrm{d} k_y \; \\frac{1}{2} |k_y \\phi(k_y) |^2 }
    $$
    """
    E_V_ky = get_energy_V_ky(p, dx=dx)
    E_V = np.mean(E_V_ky, axis=-1)
    return E_V


# Phase Angle Spectra


def get_delta_ky(n: np.ndarray, p: np.ndarray, real=True) -> np.ndarray:
    n_dft = np.fft.fft2(n, norm="ortho")
    p_dft = np.fft.fft2(p, norm="ortho")
    delta_k = np.imag(np.log(np.conjugate(n_dft) * p_dft))
    delta_k = np.mean(delta_k, axis=-1)  # mean in x
    # Get Real component
    delta_k = delta_k[..., : n.shape[1] // 2]
    return delta_k
