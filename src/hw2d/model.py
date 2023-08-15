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
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Local Imports
from hw2d.initializations.fourier_noise import get_fft_noise
from hw2d.utils.namespaces import Namespace

# NumPy Version
from hw2d.arakawa.numpy_arakawa import periodic_arakawa_vec

periodic_arakawa = periodic_arakawa_vec
from hw2d.gradients.numpy_gradients import periodic_laplace_N, periodic_gradient
from hw2d.poisson_solvers.numpy_fourier_poisson import fourier_poisson_double
from hw2d.physical_properties.numpy_properties import *

try:
    # Numba
    from hw2d.arakawa.numba_arakawa import periodic_arakawa_stencil
    from hw2d.gradients.numba_gradients import periodic_laplace_N  # , periodic_gradient

    # from hw2d.poisson_solvers.numba_fourier_poisson import fourier_poisson_double

    periodic_arakawa = periodic_arakawa_stencil
except:
    pass
# Other
# from hw2d.poisson_solvers.pyfftw_fourier_poisson import fourier_poisson_pyfftw


class HW:
    def __init__(
        self,
        dx: float,
        N: int,
        c1: float,
        nu: float,
        k0: float,
        arakawa_coeff: float = 1,
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
            arakawa_coeff (float, optional): Coefficient for the Arakawa scheme. Default is 1.
            kappa_coeff (float, optional): Coefficient for d/dy phi. Default is 1.
            debug (bool, optional): Flag to enable debugging mode. Default is False.
            TEST_CONSERVATION (bool, optional): Flag to test conservation properties. Default is True.
        """
        # Numerical Schemes
        self.poisson_solver = fourier_poisson_double
        self.diffuse_N = periodic_laplace_N
        self.arakawa = periodic_arakawa
        self.gradient_func = periodic_gradient
        # Physical Values
        self.N = int(N)
        self.c1 = c1
        self.nu = (-1) ** (self.N + 1) * nu
        self.k0 = k0
        self.arakawa_coeff = arakawa_coeff
        self.kappa_coeff = kappa_coeff
        self.dx = dx
        self.L = 2 * np.pi / k0
        # Physical Properties
        self.TEST_CONSERVATION = TEST_CONSERVATION
        # Debug Values
        self.debug = debug
        self.counter = 0
        self.watch_fncs = (
            "rk4_step",
            "euler_step",
            "get_phi",
            "diffuse",
            "gradient_2d",
            "arakawa",
        )
        self.timings = {k: 0 for k in self.watch_fncs}
        self.calls = {k: 0 for k in self.watch_fncs}

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
        df["%time"] = df["time"] / df["time"]["rk4_step"] * 100
        df.sort_values("time/call", inplace=True)
        print(df)

    def euler_step(self, plasma: Namespace, dt: float, dx: float) -> Namespace:
        t0 = time.time()
        d = dt * self.gradient_2d(plasma=plasma, phi=plasma["phi"], dt=0, dx=dx)
        y = plasma + d
        y["phi"] = self.get_phi(omega=y.omega, dx=dx)
        y["age"] = plasma.age + dt
        self.log("euler_step", time.time() - t0)
        return y

    def rk4_step(self, plasma: Namespace, dt: float, dx: float) -> Namespace:
        # RK4
        t0 = time.time()
        yn = plasma
        # pn = self.get_phi(omega=yn.omega, dx=dx)  # TODO: only execute for t=0
        pn = yn.phi
        k1 = dt * self.gradient_2d(plasma=yn, phi=pn, dt=0, dx=dx)
        p1 = self.get_phi(omega=(yn + k1 * 0.5).omega, dx=dx)
        k2 = dt * self.gradient_2d(plasma=yn + k1 * 0.5, phi=p1, dt=dt / 2, dx=dx)
        p2 = self.get_phi(omega=(yn + k2 * 0.5).omega, dx=dx)
        k3 = dt * self.gradient_2d(plasma=yn + k2 * 0.5, phi=p2, dt=dt / 2, dx=dx)
        p3 = self.get_phi(omega=(yn + k3).omega, dx=dx)
        k4 = dt * self.gradient_2d(plasma=yn + k3, phi=p3, dt=dt, dx=dx)
        # p4 = self.get_phi(k4.omega)
        # TODO: currently adds two timesteps
        y1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) * (1 / 6)
        phi = self.get_phi(omega=y1.omega, dx=dx)
        self.log("rk4_step", time.time() - t0)
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
        # Set properties not valid through y1
        y1["phi"] = phi
        y1["age"] = plasma.age + dt
        return y1

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
        plasma: Namespace,
        phi: np.ndarray,
        dt: float,
        dx: float,
        debug: bool = False,
    ) -> np.ndarray:
        arak_comp_o = 0
        arak_comp_n = 0
        kap = 0
        DO = 0
        Dn = 0
        t0 = time.time()
        dy_p = self.gradient_func(phi, dx=dx, axis=0)
        self.log("gradient_2d", time.time() - t0)

        # Calculate Gradients
        diff = phi - plasma.density

        # Step 2.1: New Omega.
        o = self.c1 * diff
        if self.arakawa_coeff:
            t0 = time.time()
            arak_comp_o = -self.arakawa_coeff * self.arakawa(
                zeta=phi, psi=plasma.omega, dx=dx
            )
            self.log("arakawa", time.time() - t0)
            o += arak_comp_o
        if self.nu:
            Do = self.nu * self.diffuse(arr=plasma.omega, dx=dx, N=self.N)
            o += Do

        # Step 2.2: New Density.
        n = self.c1 * diff
        if self.arakawa_coeff:
            t0 = time.time()
            arak_comp_n = -self.arakawa_coeff * self.arakawa(
                zeta=phi, psi=plasma.density, dx=dx
            )
            self.log("arakawa", time.time() - t0)
            n += arak_comp_n
        if self.kappa_coeff:
            kap = -self.kappa_coeff * dy_p
            n += kap
        if self.nu:
            Dn = self.nu * self.diffuse(arr=plasma.density, dx=dx, N=self.N)
            n += Dn

        if debug:
            print(
                "  |  ".join(
                    [
                        f"  dO/dt = {np.max(np.abs(self.c1 * diff)):>8.2g} + {np.max(np.abs(arak_comp_o)):>8.2g} + {np.max(np.abs(DO)):>8.2g}"
                        f"  dn/dt = {np.max(np.abs(self.c1 * diff)):>8.2g} + {np.max(np.abs(arak_comp_n)):>8.2g} + {np.max(np.abs(kap)):>8.2g} + {np.max(np.abs(Dn)):>8.2g}",
                        f"  dO/dt = {np.mean(self.c1 * diff):>8.2g} + {np.mean(arak_comp_o):>8.2g} + {np.mean(DO):>8.2g}",
                        f"  dn/dt = {np.mean(self.c1 * diff):>8.2g} + {np.mean(arak_comp_n):>8.2g} + {np.mean(kap):>8.2g} + {np.mean(Dn):>8.2g}",
                    ]
                )
            )

        return_dict = Namespace(
            density=n,
            omega=o,
            phi=phi,  # NOTE: NOT A GRADIENT
            age=plasma.age + dt,
            dx=dx,
        )

        if self.TEST_CONSERVATION:
            gamma_n, gamma_c = self.get_gammas(plasma.density, phi)
            Dp = self.nu * self.diffuse(arr=phi, dx=dx, N=self.N)
            DE = get_DE(n=plasma.density, p=phi, Dn=Dn, Dp=Dp)
            DU = get_DU(n=plasma.density, o=plasma.omega, Dn=Dn, Dp=Dp)
            dE_dt = get_dE_dt(gamma_n=gamma_n, gamma_c=gamma_c, DE=DE)
            dU_dt = get_dU_dt(gamma_n=gamma_n, DU=DU)
            return_dict["dE"] = dE_dt
            return_dict["dU"] = dU_dt

        return return_dict

    def get_gammas(self, n: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Generate Energy Gradients
        gamma_n = get_gamma_n(n=n, p=p, dx=self.dx)
        gamma_c = get_gamma_c(n=n, p=p, c1=self.c1, dx=self.dx)
        return gamma_n, gamma_c
