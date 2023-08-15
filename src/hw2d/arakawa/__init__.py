"""
The `hw2d.arakawa` Module
=========================

This module provides implementations of the Arakawa scheme to compute the Poisson bracket. 
The Arakawa scheme ensures accurate computation of the Poisson bracket, avoiding spurious oscillations which can arise from naive discretization.

The module offers implementations based on different computational frameworks:

- **NumPy**: Efficient array operations and integration with the scientific Python ecosystem.
- **Numba**: Just-in-time compiled functions optimized for speed.

To see specific functions and their documentation, navigate to the respective sub-modules:

- `hw2d.arakawa.numpy_arakawa`: Arakawa scheme functions implemented using NumPy.
- `hw2d.arakawa.numba_arakawa`: Arakawa scheme functions accelerated with Numba.
"""

# # arakawa_scheme/__init__.py

# from . import numpy_arakawa

# try:
#     from . import numba_arakawa
#     _NUMBA_AVAILABLE = True
#     _USE_NUMBA = True  # Default to use Numba if available
# except ImportError:
#     _NUMBA_AVAILABLE = False
#     _USE_NUMBA = False  # Fallback to not using Numba

# def use_numba(enable=True):
#     """Enable or disable Numba-accelerated functions for Arakawa scheme."""
#     global _USE_NUMBA
#     if enable and not _NUMBA_AVAILABLE:
#         raise ImportError("Numba is not available on this system for Arakawa scheme.")
#     _USE_NUMBA = enable

# def periodic_arakawa(*args, **kwargs):
#     if _USE_NUMBA:
#         return numba_arakawa.periodic_arakawa_st2(*args, **kwargs)
#     else:
#         return numpy_arakawa.periodic_arakawa_vec(*args, **kwargs)
