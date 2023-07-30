import numpy as np
import matplotlib.pyplot as plt
import time
# Local Imports
from hw2d.initializations.fourier_noise import get_fft_noise
from hw2d.utils.namespaces import Namespace
# NumPy
from hw2d.arakawa.numpy_arakawa import periodic_arakawa_vec
periodic_arakawa = periodic_arakawa_vec
from hw2d.gradients.numpy_gradients import periodic_laplace_N, periodic_gradient
from hw2d.poisson_solvers.numpy_fourier_poisson import fourier_poisson_double


class HW:
    def __init__(
        self,
        dx,
        N,
        c1,
        nu,
        k0,
        arakawa_coeff=1,
        kappa_coeff=1,
        debug=False,
    ):
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
        # Debug Values
        self.debug = debug
        self.counter = 0
        self.watch_fncs = ("rk4_step", "get_phi", "diffuse", "np_gradient", "arakawa")
        self.timings = {k:0 for k in self.watch_fncs}
        self.calls = {k:0 for k in self.watch_fncs}

    def log(self, name, time):
        self.timings[name] += time
        self.calls[name] += 1

    def print_log(self):
        import pandas as pd
        df = pd.DataFrame({"time": self.timings, "calls": self.calls})
        df["time/call"] = df["time"] / df["calls"]
        df["%time"] = df["time"] / df["time"]["rk4_step"] * 100
        df.sort_values('time/call', inplace=True)
        print(df)

    def euler_step(self, yn, dt):
        d = dt * self.gradient_2d(yn, pn, dt=0)
        y = yn + d
        phi = self.get_phi(y)
        t1 = time.time()
        return Namespace(
            density=y.density,
            omega=y.omega,
            phi=phi,
            age=yn.age + dt,
        )

    def rk4_step(self, plasma, dt, dx):
        # RK4
        t0 = time.time()
        yn = plasma
        #pn = self.get_phi(plasma=yn, dx=dx)  # TODO: only execute for t=0
        pn = yn.phi
        k1 = dt * self.gradient_2d(plasma=yn, phi=pn, dt=0, dx=dx)
        p1 = self.get_phi(plasma=yn + k1 * 0.5, dx=dx)
        k2 = dt * self.gradient_2d(plasma=yn + k1 * 0.5, phi=p1, dt=dt / 2, dx=dx)
        p2 = self.get_phi(plasma=yn + k2 * 0.5, dx=dx)
        k3 = dt * self.gradient_2d(plasma=yn + k2 * 0.5, phi=p2, dt=dt / 2, dx=dx)
        p3 = self.get_phi(plasma=yn + k3, dx=dx)
        k4 = dt * self.gradient_2d(plasma=yn + k3, phi=p3, dt=dt, dx=dx)
        # p4 = self.get_phi(k4)
        # TODO: currently adds two timesteps
        y1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) * (1 / 6)
        phi = self.get_phi(plasma=y1, dx=dx)
        t1 = time.time()
        self.log("rk4_step", t1-t0)
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
                        f"{t1-t0:>6.02f}s",
                    ]
                )
            )
        return Namespace(
            density=y1.density,
            omega=y1.omega,
            phi=phi,
            age=plasma.age + dt,
        )

    def get_phi(self, plasma, dx):
        t0 = time.time()
        omega = plasma.omega
        o_mean = np.mean(omega)
        centered_omega = omega - o_mean
        phi = self.poisson_solver(centered_omega, dx)
        self.log("get_phi", time.time() - t0)
        return phi

    def diffuse(self, arr, dx, N):
        t0 = time.time()
        arr = self.diffuse_N(arr, dx, N)
        self.log("diffuse", time.time() - t0)
        return arr

    def gradient_2d(self, plasma, phi, dt, dx, debug=False):
        arak_comp_o = 0
        arak_comp_n = 0
        kap = 0
        DO = 0
        Dn = 0
        t0 = time.time()
        dy_p = self.gradient_func(phi, dx=dx)[0]
        self.log("np_gradient", time.time() - t0)

        # Calculate Gradients
        diff = phi - plasma.density

        # Step 2.1: New Omega.
        o = self.c1 * diff
        if self.arakawa_coeff:
            t0 = time.time()
            arak_comp_o = -self.arakawa_coeff * self.arakawa(phi, plasma.omega, dx)
            self.log("arakawa", time.time() - t0)
            o += arak_comp_o
        if self.nu:
            DO = self.nu * self.diffuse(plasma.omega, dx, self.N)
            o += DO

        # Step 2.2: New Density.
        n = self.c1 * diff
        if self.arakawa_coeff:
            t0 = time.time()
            arak_comp_n = -self.arakawa_coeff * self.arakawa(
                phi, plasma.density, dx
            )
            self.log("arakawa", time.time() - t0)
            n += arak_comp_n
        if self.kappa_coeff:
            kap = -self.kappa_coeff * dy_p
            n += kap
        if self.nu:
            Dn = self.nu * self.diffuse(plasma.density, dx, self.N)
            n += Dn
        
        if debug:
            print("  |  ".join([
                f"  dO/dt = {np.max(np.abs(self.c1 * diff)):>8.2g} + {np.max(np.abs(arak_comp_o)):>8.2g} + {np.max(np.abs(DO)):>8.2g}"
                f"  dn/dt = {np.max(np.abs(self.c1 * diff)):>8.2g} + {np.max(np.abs(arak_comp_n)):>8.2g} + {np.max(np.abs(kap)):>8.2g} + {np.max(np.abs(Dn)):>8.2g}",
                f"  dO/dt = {np.mean(self.c1 * diff):>8.2g} + {np.mean(arak_comp_o):>8.2g} + {np.mean(DO):>8.2g}",
                f"  dn/dt = {np.mean(self.c1 * diff):>8.2g} + {np.mean(arak_comp_n):>8.2g} + {np.mean(kap):>8.2g} + {np.mean(Dn):>8.2g}"
            ]))

        return Namespace(
            density=n,
            omega=o,
            phi=phi,  # NOTE: NOT A GRADIENT
            age=plasma.age + dt,
            dx=dx,
        )

