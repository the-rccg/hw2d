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
