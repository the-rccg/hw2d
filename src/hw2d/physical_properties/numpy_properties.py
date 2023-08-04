# Define Frame Properties
# Assume (..., y, x) as shape
import numpy as np
from typing import Dict, Tuple
from hw2d.utils.namespaces import Namespace
from hw2d.gradients.numpy_gradients import periodic_laplace_N, periodic_gradient


# Gammas


def get_gamma_n(n: np.ndarray, p: np.ndarray, dx: float, dy_p=None) -> float:
    """
    Average Particle Flux

    $Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}$
    """
    if dy_p is None:
        dy_p = periodic_gradient(p, dx=dx, axis=-2)
    gamma_n = -np.mean((n * dy_p), axis=(-1, -2))
    return gamma_n


def get_gamma_c(n: np.ndarray, p: np.ndarray, c1: float, dx: float) -> float:
    """Gamma_c = c_1 \int{d^2 x (\tilde{n} - \tilde{\phi})^2"""
    gamma_c = c1 * np.mean((n - p) ** 2, axis=(-1, -2))
    return gamma_c


# Spectral Gamma_n


def get_gamma_n_ky(
    n: np.ndarray, p: np.ndarray, dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the spectral components of Gamma_n"""
    ky_max = n.shape[-2] // 2
    n_dft = np.fft.fft2(n, norm="ortho")  # n(ky, kx)
    p_dft = np.fft.fft2(p, norm="ortho")  # p(ky, kx)
    k_kx, k_ky = np.meshgrid(
        *[np.fft.fftfreq(int(i), d=dx) * 2 * np.pi for i in n.shape]
    )  # k(ky, kx)
    gamma_n_k = n_dft * 1j * k_ky * np.conjugate(p_dft)  # gamma_n(ky, kx)
    integrated_gamma_n_k = np.mean(np.real(gamma_n_k), axis=-1)[:ky_max]  # gamma_n(ky)
    ky = k_ky[:ky_max, 0]
    return ky, integrated_gamma_n_k


def get_gamma_n_spectrally(n: np.ndarray, p: np.ndarray, dx: float) -> float:
    ky, integrated_gamma_n_k = get_gamma_n_ky(n=n, p=p, dx=dx)
    gamma_n = np.mean(integrated_gamma_n_k, axis=-1)  # gamma_n
    return gamma_n


# Energies


def get_energy(n: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """Energy of the HW2D system
    $$ E = \frac{1}{2} \int{d^2 x (\tilde{n}^2 + |\nabla_\bot\tilde{\phi}|^2)} $$
    """
    grad_phi = periodic_laplace_N(phi, dx=dx, N=1)
    # Norm
    norm_grad_phi = np.abs(grad_phi)
    # Integrate, then divide by 2
    integral = np.mean((n**2) + (norm_grad_phi**2), axis=(-1, -2))
    return integral / 2


def get_enstrophy(n: np.ndarray, omega: np.ndarray, dx: float) -> np.ndarray:
    """Enstrophy of the HW2D system
    $$ U = \frac{1}{2} \int{d^2 x (\tilde{n}^2 - \nabla^2_\bot \tilde{\phi})^2} = \frac{1}{2} \int{d^2 x (\tilde{n}-\tilde{\Omega})^2} $$
    """
    omega = omega - np.mean(omega, axis=(-1, -2), keepdims=True)
    integral = np.mean(((n - omega) ** 2), axis=(-1, -2))
    return integral / 2


def get_enstrophy_phi(n: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """Enstrophy of the HW2D system from phi
    $$ U = \frac{1}{2} \int{d^2 x (\tilde{n}^2 - \nabla^2_\bot \tilde{\phi})^2} = \frac{1}{2} \int{d^2 x (\tilde{n}-\tilde{\Omega})^2} $$
    """
    omega = periodic_laplace_N(phi, dx, N=1)
    omega -= np.mean(omega, axis=(-1, -2), keepdims=True)
    integral = np.mean(((n - omega) ** 2), axis=(-1, -2))
    return integral / 2
