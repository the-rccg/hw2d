import numpy as np
import h5py
import os
import pytest

from hw2d.utils.namespaces import Namespace
from hw2d.gradients.numpy_gradients import *
from hw2d.physical_properties.numpy_properties import *
from hw2d.model import HW  # (grid_size, L)


atol = 1e-5
rtol = 1e-5


def test_energy_conservation(dt: float = 0.0001):
    # Initial State
    with h5py.File(
        f"{os.path.dirname(os.path.realpath(__file__))}/reference.h5"
    ) as reference:
        plasma = Namespace(
            density=reference["density"][0].astype(np.float64),
            phi=reference["phi"][0].astype(np.float64),
            omega=reference["omega"][0].astype(np.float64),
            age=0,
        )
        props = dict(reference.attrs)
    # Make things easier to write
    dx = props["dx"]
    nu = props["nu"]
    N = props["N"]
    # Define Physics
    hw = HW(
        dx=dx,
        N=N,
        nu=nu,
        c1=props["c1"],
        k0=props["k0"],
        arakawa_coeff=props["arakawa_coeff"],
        kappa_coeff=props["kappa_coeff"],
    )
    plasma["phi"] = hw.get_phi(omega=plasma["omega"], dx=dx)
    # Get initial values
    n = plasma["density"]
    p = plasma["phi"]
    o = plasma["omega"]
    E = get_energy(n=n, phi=p, dx=dx)
    U = get_enstrophy(n=n, omega=o, dx=dx)
    # Simulate a few timesteps to verify
    print(
        f"{'E absolute':>10} ({'E relative':>10})   {'U absolute':>10} ({'U relative':>10})"
    )
    for _ in range(3):
        # Generate Energy Gradients
        gamma_n = get_gamma_n(n=n, p=p, dx=dx)
        gamma_c = get_gamma_c(n=n, p=p, c1=props["c1"], dx=dx)
        Dn = nu * hw.diffuse(n, N=N, dx=dx)
        Dp = nu * hw.diffuse(p, N=N, dx=dx)
        DE = get_DE(n=n, p=p, Dn=Dn, Dp=Dp)
        DU = get_DU(n=n, o=o, Dn=Dn, Dp=Dp)
        dE_dt = get_dE_dt(gamma_n=gamma_n, gamma_c=gamma_c, DE=DE)
        dU_dt = get_dU_dt(gamma_n=gamma_n, DU=DU)
        # Stepping for Energies
        E_exp = E + dt * dE_dt
        U_exp = U + dt * dU_dt
        # Generate Previous Energies
        E_prev = E
        U_prev = U
        # Stepping for Plasma State
        # plasma = hw.rk4_step(plasma, dt=dt, dx=dx)
        plasma = plasma + dt * hw.gradient_2d(
            plasma=plasma, phi=plasma["phi"], dt=0, dx=dx
        )
        plasma["phi"] = hw.get_phi(omega=plasma["omega"], dx=dx)
        # New
        n = plasma["density"]
        p = plasma["phi"]
        o = plasma["omega"]
        E = get_energy(n=n, phi=p, dx=dx)
        U = get_enstrophy(n=n, omega=o, dx=dx)
        assert np.isclose(E_exp, E, atol=atol, rtol=rtol)
        assert np.isclose(U_exp, U, atol=atol, rtol=rtol)
        print(
            f"{E-E_exp:>10.2e} ({(E-E_exp)/E:>10.2e})   {U-U_exp:>10.2e} ({(U-U_exp)/U:>10.2e})"
        )


if __name__ == "__main__":
    pytest.main()
