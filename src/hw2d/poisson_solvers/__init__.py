"""
The `hw2d.poisson_solvers` Module
=================================

This module provides a suite of solvers for the Poisson equation, which is a fundamental equation in fluid dynamics and numerical simulations. Given the gradient (or Laplace) of a field, these solvers can retrieve the original field. They are crucial for inverse operations in spectral methods and other numerical techniques.

The solvers are organized based on their computational framework and method:

- **NumPy**: Solvers that leverage the NumPy library for efficient array operations.
- **Numba**: Solvers that utilize the Numba JIT compiler for enhanced performance.
- **Fourier-based**: Solvers that leverage the Fourier transform for solving the Poisson equation in spectral space.

To explore the specific functions and their documentation, navigate to the respective sub-modules:

- `hw2d.poisson_solvers.numpy_fourier_poisson`: Fourier-based Poisson solvers using NumPy.
- `hw2d.poisson_solvers.numba_fourier_poisson`: Fourier-based Poisson solvers optimized with Numba.

Select the appropriate solver based on your application's requirements and available computational resources.
"""
