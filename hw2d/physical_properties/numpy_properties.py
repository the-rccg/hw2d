# Define Frame Properties
# Assume (..., y, x) as shape
import numpy as np
from typing import Dict, Tuple
from hw2d.utils.namespaces import Namespace
from hw2d.gradients.numpy_gradients import periodic_laplace_N, periodic_gradient


def get_wavenumber(arr: np.ndarray, dx: float) -> np.ndarray:
    f = np.fft.fftfreq(n=arr, d=dx)
    k = 2 * np.pi * f
    return k


# Gammas


def get_gamma_n_kyi(
    n: np.ndarray, p: np.ndarray, dx: float, ky: np.ndarray
) -> np.ndarray:
    # DFT in Y
    n_y_dft = np.fft.fft(n, axis=0)
    p_y_dft = np.fft.fft(p, axis=0)
    gamma_n_kyi = n_y_dft * 1j * ky * np.conjugate(p_y_dft)
    return gamma_n_kyi


def integrate_gamma_n_kyi(
    gamma_n_kyi: np.ndarray, ky: np.ndarray, dx: float
) -> np.ndarray:
    """Integration in the fourier space as sum of fourier series"""
    integrated_gamma_n_kyi = np.mean(gamma_n_kyi, axis=-1)  # Mean over x
    # Integrate over y, adjust for fft
    integrated_gamma_n_kyi = (
        np.mean(integrated_gamma_n_kyi, axis=-1) * len(ky) ** 3 / dx**2
    )
    integrated_gamma_n = np.abs(integrated_gamma_n_kyi)
    return integrated_gamma_n


def get_gamma_n_spectrally(n: np.ndarray, p: np.ndarray, dx: float) -> float:
    """Mittlere Teilchen Transport
    Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}"""
    ky = get_wavenumber(n.shape[-2], dx=dx)
    gamma_n_kyi = get_gamma_n_kyi(n=n, p=p, dx=dx, ky=ky)
    gamma_n_spectral = integrate_gamma_n_kyi(gamma_n_kyi, ky, dx)
    return gamma_n_spectral


def get_gamma_n(n: np.ndarray, p: np.ndarray, dx: float, dy_p=None) -> float:
    """Mittlere Teilchen Transport
    Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}"""
    if dy_p is None:
        dy_p = periodic_gradient(p, dx=dx)
    # gamma_n = -field.integrate((n * dy_p), n.box) / n.bounds.volume
    gamma_n = -np.mean((n * dy_p)) * dx**2 / np.product(n.shape)
    return gamma_n


def get_gamma_c(n: np.ndarray, p: np.ndarray, c1: float, dx: float) -> float:
    """Gamma_c = c_1 \int{d^2 x (\tilde{n} - \tilde{\phi})^2"""
    # gamma_c = c1 * field.integrate(((n - p) ** 2), n.box)
    gamma_c = c1 * np.sum((n - p) ** 2) * dx**2
    return gamma_c


def get_gammas(
    n: np.ndarray, p: np.ndarray, c1: float, dx: float, dy_p: np.ndarray = None
) -> Tuple[float, float]:
    """
    Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}
    Gamma_c = c_1 \int{d^2 x (\tilde{n} - \tilde{\phi})^2
    """
    if dy_p is None:
        dy_p = periodic_gradient(p)
    # gamma_n = -field.integrate((n * dy_p), n.box) / n.bounds.volume
    gamma_n = -np.sum((n * dy_p)) * dx**2
    # gamma_c = c1 * field.integrate(((n - p) ** 2), n.box)
    gamma_c = c1 * np.sum((n - p) ** 2) * dx**2
    return gamma_n, gamma_c


# Angle stuff


def adjust_scale(values: np.ndarray) -> np.ndarray:
    values[np.isfinite(values)] %= 2 * np.pi
    values[values > np.pi] -= 2 * np.pi
    return values


def get_angle(dft: Dict[str, np.ndarray]) -> np.ndarray:
    angle_phi = np.angle(dft["phi"])
    angle_n = np.angle(dft["density"])
    angle = angle_phi - angle_n
    return angle


def phase_angle_stats(
    c1: float,
    k0: float,
    angle: float,
    time_dim: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Dict[str, float]:
    angle = adjust_scale(angle)
    k_max_idx = int(angle.shape[-2] / 2)

    def cut(tensor):
        return tensor[-2][:k_max_idx]

    # mean
    k_mean = cut(np.mean(np.mean(angle, axis=dim2), axis=time_dim))
    # std
    k_std = cut(np.std(angle, axis=[time_dim, dim2]))
    # median
    k_median = cut(np.median(angle, axis=[time_dim, dim2]))
    # theoretical
    k_theo = np.wrap(
        [
            get_omega_Re(c1=c1, ky=k_idx * k0, kx=0)
            for k_idx in range(int(angle.dimension(dim1).size / 2))
        ],
        spatial(y=int(angle.shape[dim1] / 2)),
    )
    # Losses to Theoretical
    k_stats = {
        "k_mean": k_mean,
        "k_std": k_std,
        "k_median": k_median,
        "k_theo": k_theo,
    }
    return k_stats


# Phase Angle Spectra


def get_delta_ky(data: np.ndarray) -> np.ndarray:
    fields = [batch("time"), spatial("y"), spatial("x")]
    field_list = ["density", "phi"]
    field_data = {
        field_name: np.tensor(data[field_name], *fields) for field_name in field_list
    }
    dft = {
        field_name: np.fft(np.mean(field_data[field_name].data, dim="x"), dims="y")
        for field_name in field_data.keys()
    }
    delta_k = np.imag(np.log(np.conjugate(dft["density"]) * dft["phi"]))
    return delta_k.numpy(["time", "y"])


def get_delta_ky_dft(dft_density: np.ndarray, dft_phi: np.ndarray) -> np.ndarray:
    return np.imag(np.log(np.conjugate(dft_density) * dft_phi))


# Energies


def get_energy(n: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    $$ E = \frac{1}{2} \int{d^2 x (\tilde{n}^2 + |\nabla_\bot\tilde{\phi}|^2)} $$
    """
    # nabla_bot phi
    # grad_phi = field.spatial_gradient(phi, stack_dim=channel("gradient"))
    grad_phi = periodic_laplace_N(phi, dx, N=1)
    grad_phi = np.sum(phi, dim=channel("gradient"))
    # grad_phi = np.sum(
    #     np.vec_squared(phi.values, vec_dim=channel("gradients")),
    #     dim=channel("gradients"),
    # )
    # Norm
    norm_grad_phi = np.abs(grad_phi)
    # Integrate, then divide by 2
    # Equivalent to: np.sum((n ** 2).values + (norm_grad_phi ** 2), dim="y,x") * phi.dx[0] ** 2
    # integral = field.integrate((n ** 2) + (norm_grad_phi ** 2), n.box)
    integral = field.sample((n**2) + (norm_grad_phi**2), n.box)
    return integral / 2


def get_enstrophy(n: np.ndarray, omega: np.ndarray, dx: float) -> np.ndarray:
    """
    $$ U = \frac{1}{2} \int{d^2 x (\tilde{n}^2 - \nabla^2_\bot \tilde{\phi})^2} = \frac{1}{2} \int{d^2 x (\tilde{n}-\tilde{\Omega})^2} $$
    """
    omega -= np.mean(omega, axis=[-1, -2])
    integral = np.sum(((n - omega) ** 2), axis=[-1, -2]) * dx**2
    # integral = field.integrate((n - omega) ** 2, n.box)
    # integral = field.sample((n - omega) ** 2, n.box)
    return integral / 2


def get_enstrophy_phi(n: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    $$ U = \frac{1}{2} \int{d^2 x (\tilde{n}^2 - \nabla^2_\bot \tilde{\phi})^2} = \frac{1}{2} \int{d^2 x (\tilde{n}-\tilde{\Omega})^2} $$
    """
    omega = periodic_laplace_N(phi, dx, N=1)
    omega -= np.mean(omega.data, dim="x,y")
    # integral = field.integrate((n - omega) ** 2, n.box)
    integral = np.sum(((n - omega) ** 2), axis=[-1, -2]) * dx**2
    # integral = field.sample((n - omega) ** 2, n.box)
    return integral / 2


def gradient_invariants(
    gamma_n: np.ndarray,
    gamma_c: np.ndarray,
    n: np.ndarray,
    o: np.ndarray,
    p: np.ndarray,
    N: int,
    nu: float,
    dt: float,
    dx: float,
):
    """returns dE, dU for timestepping
    Equations:
    dE/dt = G_n + G_c - D^E
    dU/dt = G_n - D^U
    """
    # o -= np.mean(o.data, dim="x,y")
    # dE/dt = G_n + G_c - D^E
    # D^E = integral(dx^2 (n*D^n - p*D^p))
    D_n = (-1) ** (N + 1) * nu * periodic_laplace_N(n, dx, N)
    D_p = (-1) ** (N + 1) * nu * periodic_laplace_N(o, dx, N)
    DE = -field.sample((n * D_n - p * D_p), n.box)
    DEn = -field.sample(n * D_n, n.box)
    DEv = field.sample(p * D_p, n.box)
    # dU/dt = G_n - G_c - D^U
    # D^U = -integral(dx^2 (n-O)*(D^n - D^p))
    DU = -field.sample((n - o) * (D_n - D_p), n.box)
    # dE/dt = G_n - G_c - DE
    dE = gamma_n - gamma_c - DE  # * dt
    # dU/dt = G_n - DU
    dU = gamma_n - DU
    return dE, dU, DE, DU, DEn, DEv


def mean_gradient_invariants(
    gamma_n: float,
    gamma_c: float,
    n: np.ndarray,
    o: np.ndarray,
    p: np.ndarray,
    N: int,
    nu: float,
    dx: float,
    D_n=None,
    D_p=None,
) -> Tuple[float, float]:
    """returns dE, dU for timestepping
    Equations:
    dE/dt = G_n + G_c - D^E
    dU/dt = G_n - D^U

    discretized using np.sum to get scalar of 2D
    """
    # dE/dt = G_n + G_c - D^E
    # D^E = integral(dx^2 (n*D^n - p D^p))
    if D_n is None:
        D_n = nu * periodic_laplace_N(n, dx, N)
    if D_p is None:
        D_p = nu * periodic_laplace_N(o, dx, N)
    # DE = field.integrate(
    # TODO: WHY THE MINUS SIGN?
    DE = -field.sample((n * D_n - p * D_p), n.bounds)
    # dU/dt = G_n - D^U
    # D^U = -integral(dx^2 (n-O)*(D^n - D^p))
    # DU = -field.integrate(
    DU = -field.sample((n - o) * (D_n - D_p), n.bounds)
    # dE/dt = G_n - G_c - DE
    dE = gamma_n - gamma_c - DE
    # dU/dt = G_n - DU
    dU = gamma_n - DU
    return dE, dU


# Spectral Energies


def calculate_E_v_k(phi: np.ndarray, dx: float) -> np.ndarray:
    phi_dft = make_kx_ky(phi)
    kx = get_wavenumber(phi.shape[2], dx[1])
    ky = get_wavenumber(phi.shape[1], dx[0])
    return (kx**2 + ky**2) * np.abs(phi_dft) ** 2 / 2


def calculate_E_v_ky(phi_dft: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return (2 * ky**2) * np.abs(phi_dft) ** 2 / 2


def calculate_E_n_k(n: np.ndarray) -> np.ndarray:
    n_dft = make_kx_ky(n)
    return np.abs(n_dft) ** 2 / 2


def calculate_E_n_ky(n_dft: np.ndarray) -> np.ndarray:
    return np.abs(n_dft) ** 2 / 2
