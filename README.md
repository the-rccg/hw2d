# The Hasegawa-Wakatani model of plasma turbulence

This repository contains a reference implementations for the Hasegawa-Wakatani model in two dimensions using Python.
The purpose is to provide a playground for education and scientific purposes: be it testing numerical or machine learning methods, or building related models quicker.

Stable, verified parameters will be published with this repository.

Install a pure NumPy version via 
```pip install hw2d```
and to include accelerators like numba, use the following:
```pip install hw2d[accelerators]```

The implementation presented here is by no means meant to be the optimal, but an easy to understand starting point to build bigger things upon and serve as a reference for other work.
This reference implementation uses:
- Gradients: Central finite difference schemes (2nd order accurate)
- Poisson Bracket: Arakawa Scheme (2nd order accurate, higher order preserving)
- Poisson Solver: Fourier based solfer
- Time Integration: Explicit Runge Kutte (4th order accurate)
The framework presented here can be easily extended to use alternative implementations. 

Pull requests are strongly encouraged. If you don't know where to start, implementing new numerical methods or alternative accelerators make for good first projects.

## The Hasegawa-Wakatani Model

The HW model describes drift-wave turbulence using two physical fields: the density $n$ and the potential $\phi$ using various gradients on these. 

$$
\begin{align}
    \partial_t n &= c_1 \left( \phi - n \right)
                     - \left[ \phi, n \right]
                     - \kappa_n \partial_y \phi
                     - \nu \nabla^{2N} n \,,
             \\
    \partial_t \Omega &= c_1 \left( \phi - n \right)
                                      - \left[ \phi, \Omega \right]
                                      - \nu \nabla^{2N} \Omega \,.
             \\
             \Omega &= \nabla^2 \phi
\end{align}
$$
