"""
hw2d.model: Hasegawa-Wakatani 2D Simulation Module

This module provides the core functionality to simulate the Hasegawa-Wakatani (HW) model in two dimensions.
It offers flexibility in terms of numerical schemes and allows for comprehensive profiling and debugging
of the simulation process.

Functions:
    - euler_step: Implements the Euler time-stepping method.
    - rk4_step: Implements the Runge-Kutta 4th order time-stepping method.
    - get_phi: Computes the electrostatic potential from vorticity.
    - diffuse: Applies diffusion to an array.
    - gradient_2d: Computes the 2D gradient of the system.
    - get_gammas: Computes energy gradients.

Classes:
    - HW: Represents the primary simulation entity for the Hasegawa-Wakatani model.

Notes:
    The module supports both NumPy and Numba for computational operations and provides detailed logging and
    profiling capabilities.
"""

from typing import Tuple, Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Local Imports
from hw2d.initializations.fourier_noise import get_fft_noise
from hw2d.utils.namespaces import Namespace

# NumPy Version
from hw2d.gradients.numpy_gradients import periodic_laplace_N, periodic_gradient
from hw2d.poisson_bracket.numpy_arakawa import periodic_arakawa_vec
from hw2d.poisson_solvers.numpy_fourier_poisson import fourier_poisson_double
from hw2d.physical_properties.numpy_properties import get_DE, get_DU, get_dE_dt, get_dU_dt, get_gamma_n, get_gamma_c

periodic_arakawa = periodic_arakawa_vec

try:
    # Numba
    from hw2d.gradients.numba_gradients import periodic_laplace_N, periodic_gradient
    from hw2d.poisson_bracket.numba_arakawa import periodic_arakawa_stencil
    from hw2d.poisson_solvers.numba_fourier_poisson import fourier_poisson_double

    periodic_arakawa = periodic_arakawa_stencil
    pass
except:
    pass

# Other
# from hw2d.poisson_solvers.pyfftw_fourier_poisson import fourier_poisson_pyfftw


def pass_func(*args, **kwargs):
    """do nothing function"""
    return


def hw2d_gradient(
    density: np.ndarray,
    omega: np.ndarray,
    dt: float,
    dx: float,
    get_phi_func: Callable,
    gradient_func: Callable,
    diffuse_func: Callable,
    poisson_bracket_func: Callable,
    get_gammas_func: Callable,
    poisson_bracket_coeff: float,
    kappa_coeff: float,
    c1: float,
    nu: float,
    N: int,
    log: Callable = pass_func,
    phi: np.ndarray or None = None,
    debug: bool = False,
    TEST_CONSERVATION: bool = False,
    **kwargs
) -> Namespace:
    if phi is None:
        phi = get_phi_func(omega, dx=dx)
    # Setup
    arak_comp_o = 0
    arak_comp_n = 0
    kap = 0
    Do = 0
    Dn = 0
    t0 = time.time()
    dy_p = gradient_func(phi, dx=dx, axis=0)
    log("gradient_2d", time.time() - t0)

    # Calculate Gradients
    diff = phi - density

    # Step 2.1: New Omega.
    o = c1 * diff
    if poisson_bracket_coeff:
        t0 = time.time()
        arak_comp_o = -poisson_bracket_coeff * poisson_bracket_func(
            zeta=phi, psi=omega, dx=dx
        )
        log("poisson_bracket", time.time() - t0)
        o += arak_comp_o
    if nu:
        Do = nu * diffuse_func(arr=omega, dx=dx, N=N)
        o += Do

    # Step 2.2: New Density.
    n = c1 * diff
    if poisson_bracket_coeff:
        t0 = time.time()
        arak_comp_n = -poisson_bracket_coeff * poisson_bracket_func(
            zeta=phi, psi=density, dx=dx
        )
        log("poisson_bracket", time.time() - t0)
        n += arak_comp_n
    if kappa_coeff:
        kap = -kappa_coeff * dy_p
        n += kap
    if nu:
        Dn = nu * diffuse_func(arr=density, dx=dx, N=N)
        n += Dn

    # Print gradients to see which one explodes
    if debug:
        print(
            "  |  ".join(
                [
                    f"  dO/dt = {np.max(np.abs(c1 * diff)):>8.2g} + {np.max(np.abs(arak_comp_o)):>8.2g} + {np.max(np.abs(Do)):>8.2g}"
                    f"  dn/dt = {np.max(np.abs(c1 * diff)):>8.2g} + {np.max(np.abs(arak_comp_n)):>8.2g} + {np.max(np.abs(kap)):>8.2g} + {np.max(np.abs(Dn)):>8.2g}",
                    f"  dO/dt = {np.mean(c1 * diff):>8.2g} + {np.mean(arak_comp_o):>8.2g} + {np.mean(Do):>8.2g}",
                    f"  dn/dt = {np.mean(c1 * diff):>8.2g} + {np.mean(arak_comp_n):>8.2g} + {np.mean(kap):>8.2g} + {np.mean(Dn):>8.2g}",
                ]
            )
        )

    # Structure return
    return_dict = Namespace(
        density=n,
        omega=o
    )

    if "age" in kwargs:
        return_dict["age"] = kwargs["age"] + dt

    # Add energy gradients for testing conservation
    if TEST_CONSERVATION:
        gamma_n, gamma_c = get_gammas_func(density, phi)
        Dp = nu * diffuse_func(arr=phi, dx=dx, N=N)
        DE = get_DE(n=density, p=phi, Dn=Dn, Dp=Dp)
        DU = get_DU(n=density, o=omega, Dn=Dn, Dp=Dp)
        dE_dt = get_dE_dt(gamma_n=gamma_n, gamma_c=gamma_c, DE=DE)
        dU_dt = get_dU_dt(gamma_n=gamma_n, DU=DU)
        return_dict["dE"] = dE_dt
        return_dict["dU"] = dU_dt

    return return_dict


class HW:
    def __init__(
        self,
        dx: float,
        N: int,
        c1: float,
        nu: float,
        k0: float,
        poisson_bracket_coeff: float = 1,
        kappa_coeff: float = 1,
        debug: bool = False,
        TEST_CONSERVATION: bool = True,
    ):
        """
        Initialize the Hasegawa-Wakatani (HW) simulation.

        Parameters:
            dx (float): Grid spacing.
            N (int): System size.
            c1 (float): Model-specific parameter.
            nu (float): Diffusion coefficient.
            k0 (float): Fundamental wavenumber.
            poisson_bracket_coeff (float, optional): Coefficient for the Arakawa scheme. Default is 1.
            kappa_coeff (float, optional): Coefficient for d/dy phi. Default is 1.
            debug (bool, optional): Flag to enable debugging mode. Default is False.
            TEST_CONSERVATION (bool, optional): Flag to test conservation properties. Default is True.
        """
        # Numerical Schemes
        self.poisson_solver: Callable = fourier_poisson_double
        self.diffuse_N: Callable = periodic_laplace_N
        self.poisson_bracket: Callable = periodic_arakawa
        self.gradient_func: Callable = periodic_gradient
        # Physical Values
        self.N: int = int(N)
        self.c1: float = c1
        self.nu: float = (-1) ** (self.N + 1) * nu
        self.k0: float = k0
        self.poisson_bracket_coeff: float = poisson_bracket_coeff
        self.kappa_coeff: float = kappa_coeff
        self.dx: float = dx
        self.L: float = 2 * np.pi / k0
        # Physical Properties
        self.TEST_CONSERVATION: bool = TEST_CONSERVATION
        # Debug Values
        self.debug: bool = debug
        self.counter: int = 0
        self.watch_fncs: Tuple[str] = (
            "full_step",
            "get_phi",
            "diffuse",
            "gradient_2d",
            "poisson_bracket",
        )
        self.timings: Dict[str, float] = {k: 0 for k in self.watch_fncs}
        self.calls: Dict[str, int] = {k: 0 for k in self.watch_fncs}

    def log(self, name: str, time: float):
        """
        Log the time taken by a specific function.

        Parameters:
            name (str): Name of the function.
            time (float): Time taken by the function.
        """
        self.timings[name] += time
        self.calls[name] += 1

    def print_log(self):
        """Display the timing information for the functions profiled."""
        df = pd.DataFrame({"time": self.timings, "calls": self.calls})
        df["time/call"] = df["time"] / df["calls"]
        df["%time"] = df["time"] / df["time"]["full_step"] * 100
        df.sort_values("time/call", inplace=True)
        print(df)

    def euler_step(
        self, plasma: Namespace, dt: float, dx: float, **kwargs
    ) -> Namespace:
        t0 = time.time()
        y = Namespace(**{k: v for k, v in plasma.items() if k != "phi"})
        if "phi" in plasma:
            d = dt * self.gradient_2d(**y, phi=plasma["phi"], dt=0, dx=dx)
        else:
            d = dt * self.gradient_2d(**y, dt=0, dx=dx)
        y += d
        y["phi"] = self.get_phi(omega=y.omega, dx=dx)
        y["age"] = plasma.age + dt
        # Wrap-up
        self.log("full_step", time.time() - t0)
        return y

    def rk4_step(self, plasma: Namespace, dt: float, dx: float, **kwargs) -> Namespace:
        # RK4
        t0 = time.time()
        yn = Namespace(**{k: v for k, v in plasma.items() if k != "phi"})
        # pn = self.get_phi(omega=yn.omega, dx=dx)  # TODO: only execute for t=0
        pn = plasma.phi
        k1 = dt * self.gradient_2d(**yn, phi=pn, dt=0, dx=dx)
        y1 = yn + k1 * 0.5
        p1 = self.get_phi(omega=y1.omega, dx=dx)
        k2 = dt * self.gradient_2d(**y1, phi=p1, dt=dt / 2, dx=dx)
        y2 = yn + k2 * 0.5
        p2 = self.get_phi(omega=y2.omega, dx=dx)
        k3 = dt * self.gradient_2d(**y2, phi=p2, dt=dt / 2, dx=dx)
        y3 = yn + k3
        p3 = self.get_phi(omega=y3.omega, dx=dx)
        k4 = dt * self.gradient_2d(**y3, phi=p3, dt=dt, dx=dx)
        # p4 = self.get_phi(k4.omega)
        # TODO: currently adds two timesteps
        yk1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) * (1 / 6)
        phi = self.get_phi(omega=yk1.omega, dx=dx)
        # Set properties not valid through y1
        yk1["phi"] = phi
        yk1["age"] = plasma.age + dt
        # Wrap-up
        self.log("full_step", time.time() - t0)
        if self.debug:
            print(
                " | ".join(
                    [
                        f"{plasma.age + dt:<7.04g}",
                        f"{np.max(np.abs(yn.density.data)):>7.02g}",
                        f"{np.max(np.abs(k1.density.data)):>7.02g}",
                        f"{np.max(np.abs(k2.density.data)):>7.02g}",
                        f"{np.max(np.abs(k3.density.data)):>7.02g}",
                        f"{np.max(np.abs(k4.density.data)):>7.02g}",
                        f"{time.time()-t0:>6.02f}s",
                    ]
                )
            )
        return yk1

    def get_phi(self, omega: np.ndarray, dx: float) -> np.ndarray:
        t0 = time.time()
        o_mean = np.mean(omega)
        centered_omega = omega - o_mean
        phi = self.poisson_solver(tensor=centered_omega, dx=dx)
        self.log("get_phi", time.time() - t0)
        return phi

    def diffuse(self, arr: np.ndarray, dx: float, N: int) -> np.ndarray:
        t0 = time.time()
        arr = self.diffuse_N(arr=arr, dx=dx, N=N)
        self.log("diffuse", time.time() - t0)
        return arr

    def gradient_2d(
        self,
        density: np.ndarray,
        omega: np.ndarray,
        dt: float,
        dx: float,
        phi: np.ndarray or None = None,
        debug: bool = False,
        **kwargs
    ) -> Namespace:
        return hw2d_gradient(
            density=density,
            omega=omega,
            dt=dt,
            dx=dx,
            get_phi_func=self.get_phi,
            gradient_func=self.gradient_func,
            diffuse_func=self.diffuse,
            poisson_bracket_func=self.poisson_bracket,
            get_gammas_func=self.get_gammas,
            log=self.log,
            poisson_bracket_coeff=self.poisson_bracket_coeff,
            kappa_coeff=self.kappa_coeff,
            c1=self.c1,
            nu=self.nu,
            N=self.N,
            phi=phi,
            debug=debug,
            TEST_CONSERVATION=self.TEST_CONSERVATION,
            **kwargs
        )


    def get_gammas(self, n: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Generate Energy Gradients
        gamma_n = get_gamma_n(n=n, p=p, dx=self.dx)
        gamma_c = get_gamma_c(n=n, p=p, c1=self.c1, dx=self.dx)
        return gamma_n, gamma_c
