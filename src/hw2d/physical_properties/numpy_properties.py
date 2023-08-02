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
        dy_p = periodic_gradient(p, dx=dx, axis=0)
    gamma_n = -np.mean((n * dy_p))
    return gamma_n


def get_gamma_c(n: np.ndarray, p: np.ndarray, c1: float, dx: float) -> float:
    """Gamma_c = c_1 \int{d^2 x (\tilde{n} - \tilde{\phi})^2"""
    gamma_c = c1 * np.mean((n - p) ** 2)
    return gamma_c


# Spectral Gamma_n


def get_gamma_n_kyi(
    n: np.ndarray, p: np.ndarray, dx: float, ky: np.ndarray, norm: str or None = None
) -> np.ndarray:
    """Calculate the spectral components of Gamma_n"""
    n_y_dft = np.fft.fft(n, axis=-2, norm=norm)
    p_y_dft = np.fft.fft(p, axis=-2, norm=norm)
    gamma_n_kyi = n_y_dft * 1j * ky * np.conjugate(p_y_dft)
    return gamma_n_kyi


def get_gamma_n_spectrally(n: np.ndarray, p: np.ndarray, dx: float) -> float:
    """Mittlere Teilchen Transport
    Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}"""
    ky = np.array(np.meshgrid(*[np.fft.fftfreq(int(n)) for n in n.shape]))
    gamma_n_kyi = get_gamma_n_kyi(n=n, p=p, dx=dx, ky=ky, norm=None) / ky.shape[-2]
    integrated_gamma_n_kyi = np.mean(gamma_n_kyi, axis=-1)  # Mean over x
    # Integrate over y, adjust for fft
    integrated_gamma_n_kyi = np.mean(integrated_gamma_n_kyi, axis=-1)
    integrated_gamma_n = np.real(integrated_gamma_n_kyi)
    return np.mean(integrated_gamma_n / (dx**2))


# Energies


def get_energy(n: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """Energy of the HW2D system
    $$ E = \frac{1}{2} \int{d^2 x (\tilde{n}^2 + |\nabla_\bot\tilde{\phi}|^2)} $$
    """
    grad_phi = periodic_laplace_N(phi, dx=dx, N=1)
    # Norm
    norm_grad_phi = np.abs(grad_phi)
    # Integrate, then divide by 2
    integral = np.mean((n**2) + (norm_grad_phi**2))
    return integral / 2


def get_enstrophy(n: np.ndarray, omega: np.ndarray, dx: float) -> np.ndarray:
    """Enstrophy of the HW2D system
    $$ U = \frac{1}{2} \int{d^2 x (\tilde{n}^2 - \nabla^2_\bot \tilde{\phi})^2} = \frac{1}{2} \int{d^2 x (\tilde{n}-\tilde{\Omega})^2} $$
    """
    omega -= np.mean(omega)
    integral = np.mean(((n - omega) ** 2))
    return integral / 2


def get_enstrophy_phi(n: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """Enstrophy of the HW2D system from phi
    $$ U = \frac{1}{2} \int{d^2 x (\tilde{n}^2 - \nabla^2_\bot \tilde{\phi})^2} = \frac{1}{2} \int{d^2 x (\tilde{n}-\tilde{\Omega})^2} $$
    """
    omega = periodic_laplace_N(phi, dx, N=1)
    omega -= np.mean(omega.data)
    integral = np.mean(((n - omega) ** 2))
    return integral / 2
