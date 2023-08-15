"""
The `hw2d.physical_properties` Module
=====================================

This module offers the computation of standard values for physical properties of the HW2D system.
These properties are essential for diagnostics, understanding system dynamics, and providing insights into the behavior of fluid simulations.
They include scalar values, spectral properties, and "conservation" laws.

The properties are computed using various methods:

- **NumPy**: Efficient array operations leveraging the NumPy library.
- **Numba**: JIT-compiled functions for performance-critical operations.

To explore specific functionalities:

- `hw2d.physical_properties.numpy_properties`: Properties computed using NumPy.
- `hw2d.physical_properties.numba_properties`: Properties optimized with Numba.

For detailed usage, navigate to the respective sub-modules and refer to their documentation.
"""
