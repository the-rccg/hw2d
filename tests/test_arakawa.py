from unittest import TestCase

import numpy as np
from hw2d.poisson_bracket.numpy_arakawa import periodic_arakawa as periodic_arakawa
from hw2d.poisson_bracket.numpy_arakawa import periodic_arakawa_vec as periodic_arakawa_np
from hw2d.poisson_bracket.numba_arakawa import periodic_arakawa_stencil as periodic_arakawa_nb
from hw2d.poisson_bracket.numba_arakawa import periodic_arakawa_vec as periodic_arakawa_vec_nb
from hw2d.initializations.sine import get_2d_sine


class test_arakawa(TestCase):
    def test_periodic_arakawa_nb(self):
        N = 128
        grid_size = (N, N)
        L = 1
        dx = 0.2
        input_field = get_2d_sine(grid_size, L)
        zeta = input_field.copy()
        psi = input_field.copy() / -2.0

        zeta = np.random.rand(N, N)
        psi = zeta.T #np.random.rand(N, N) / -2.0

        arr1 = periodic_arakawa(zeta, psi, dx)
        arr2 = periodic_arakawa_np(zeta, psi, dx)
        arr3 = periodic_arakawa_nb(zeta, psi, dx)
        arr4 = periodic_arakawa_vec_nb(zeta, psi, dx)
        assert np.allclose(arr1, arr2)
        assert np.allclose(arr1, arr3)
        assert np.allclose(arr1, arr4)


if __name__ == "__main__":
    test_arakawa().test_periodic_arakawa_nb()
