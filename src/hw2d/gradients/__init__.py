"""
The `hw2d.gradients` Module
===========================

This module contains a comprehensive set of functions for computing gradients, an essential operation in fluid dynamics and numerical simulations. 
These gradient functions facilitate various operations including:

- Laplace operations: Compute the second derivative of a field.
- Finite difference schemes: Approximate derivatives of functions using discrete points.
- Fourier-based gradient calculations: Use Fourier transforms for efficient and accurate gradient computations.

Implementations are organized based on the computational framework used:

- **NumPy**: Functions that leverage the NumPy library for efficient array operations.
- **Numba**: Functions that utilize the Numba JIT compiler for enhanced performance, especially suitable for large-scale simulations.

To see specific functions and their documentation, navigate to the respective sub-modules:

- `hw2d.gradients.numpy_gradients`: Gradient functions implemented using NumPy.
- `hw2d.gradients.numba_gradients`: Gradient functions accelerated with Numba.

Remember to choose the appropriate implementation based on your application's needs and the computational resources available.
"""
