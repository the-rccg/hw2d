import pytest
import h5py
import numpy as np
import os
from hw2d.physical_properties.numpy_properties import *


reference = h5py.File(f"{os.path.dirname(os.path.realpath(__file__))}/reference.h5")
n = reference["density"][0]
p = reference["phi"][0]
o = reference["omega"][0]
dx = reference.attrs["dx"]
c1 = reference.attrs["c1"]


def test_gamma_n():
    gamma_n = get_gamma_n(n=n, p=p, dx=dx)
    assert np.isclose(gamma_n, 0.57292056), gamma_n


def test_gamma_n_spectrally():
    gamma_n_spectrally = get_gamma_n_spectrally(n=n, p=p, dx=dx)
    assert np.isclose(gamma_n_spectrally, 0.5745075066507911), gamma_n_spectrally


def test_gamma_c():
    gamma_c = get_gamma_c(n=n, p=p, c1=c1, dx=dx)
    assert np.isclose(gamma_c, 0.55321026), gamma_c


def test_energy():
    energy = get_energy(n=n, phi=p, dx=dx)
    assert np.isclose(energy, 7.638421), energy


def test_enstrophy():
    enstrophy = get_enstrophy(n=n, omega=o, dx=dx)
    assert np.isclose(enstrophy, 11.743794), enstrophy


def test_enstrophy_phi():
    enstrophy_phi = get_enstrophy_phi(n=n, phi=p, dx=dx)
    assert np.isclose(enstrophy_phi, 11.453048), enstrophy_phi


if __name__ == "__main__":
    pytest.main()
