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
- Poisson Solver: Fourier based solver
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
                     - \nu \nabla^{2N} n 
             \\
    \partial_t \Omega &= c_1 \left( \phi - n \right)
                                      - \left[ \phi, \Omega \right]
                                      - \nu \nabla^{2N} \Omega 
             \\
             \Omega &= \nabla^2 \phi
\end{align}
$$

## Physical Properties

The reason why the Hasegawa-Wakatani Model has been the de-facto testing bed for new methods are its statistically stationary properties of the complex turbulent system.
The moduel includes all code needed to generate these values.
It goes further, however, and provides reference values with statistical bounds for the first time for a vast range of values.
This allows simple comparison, as well es evalutaion of new methods to one reference community built resource.

$$
\begin{align}
    \Gamma^n       \scriptstyle(x,y)\displaystyle &= -\!\! \iint{\! \mathrm{d}^2\! x \;\, n \,\partial_y \phi } \\
    \Gamma^c       \scriptstyle(x,y)\displaystyle &= c_1    \int{\! \mathrm{d}^2\! x \;\, \left(n - \phi\right)^2} \\
    \mathfrak{D}^E \scriptstyle(x,y)\displaystyle &= \quad  \int{\! \mathrm{d}^2\! x \;\, (n \mathfrak{D^n} - \phi \mathfrak{D}^\phi)} \\ 
    \mathfrak{D}^U \scriptstyle(x,y)\displaystyle &= -      \int{\! \mathrm{d}^2\! x \;\, (n - \Omega)(\mathfrak{D}^n - \mathfrak{D}^\phi)} \\
    with \quad \mathfrak{D}^n \scriptstyle(x,y)\displaystyle &= \nu \nabla^{2N} n \quad and \quad 
    \mathfrak{D}^\phi \scriptstyle(x,y)\displaystyle\; = \nu \nabla^{2N} \phi  
\end{align}
$$

Additionally, spectral properties are planned to be included, among these are:

$$
\int{\!\mathrm{d} k_y \;\, \Gamma^n\scriptstyle(k_y)} \, \displaystyle\; = -\!\! \int{\!\mathrm{d} k_y \;\, \left( i k_y \,  n\scriptstyle(k_y) \, \displaystyle\phi\scriptstyle(k_y)\displaystyle^*\right) }
$$

Note that it is the common practice across all reference texts to calculate $\int\cdot$ as $\langle \cdot \rangle$ in order to get comparable values for all properties.

