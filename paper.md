---
title: 'HW2D: A reference implementation of the Hasegawa-Wakatani model for plasma turbulence in fusion reactors'
tags:
  - Python
  - plasma physics
  - dynamics
  - fluid dynamics
  - simulation
authors:
  - name: Robin Greif
    orcid: 0000-0002-3957-2474
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Max-Planck Institute for Plasma Physics, Germany
   index: 1
 - name: Technical University Munich, Germany
   index: 2
date: 13 August 2023
bibliography: paper.bib

---

# Summary

`HW2D` is an open source reference implementation against which methods are benchmarked, 
validated, and trained on. It implements the Hasegawa-Wakatani (HW) model to simulate
drift-wave turbulence in fusion devices. Besides providing consistent numerical
simulations, it implements the physical property calculations with the normalizations 
common in reference texts that are often not clearly understandable outside of the 
community. Along with it, an iteratively expanding list of parameters are published 
of stable parameters for simulations and associated physical properties.


# Statement of need

Nuclear fusion provides the promise of clean, endless energy for humanity
in the future. One of the biggest open questions standing in the way of 
commercially viable fusion devices is confining the energy within the plasma that 
is transported out through drift wave turbulence. However, the immense complexity 
of modeling turbulence inside potential reactors still remains outside the realm 
of the biggest supercomputers. Hence, the development of new methods for simulating 
plasma turbulence is a pressing issue. 

The Hasegawa-Wakatani (HW) model has become the de-facto standard for testing methods
developed for Navier-Stokes (NS) fluids for fusion turbulence due to its similarity 
to it [@camargo]. Although first published 40 years ago [@hasegawa_wakatani], 
it is still missing an open-source reference implementation. With its rising popularity 
in other fields, especially machine learning to test limits of physics-based learning 
methods, a simple implementation is needed. A unified reference allows comparing methods 
[@Yatomi_2023;@grillix;@anderson2020elucidating;@goumiri2013reduced;@heinonen2020learning]
and reduces the overhead involved in verifying turbulent system implementations.
This verification is now automated with CI pipelines and open-sourced for the broader 
community to use.

In summary, `HW2D` provides an easy to understand, easy to use, and easy to extend 
reference implementation to help accelerate the testing of new ideas and methods.
In contrast to closed-source Fortran and C++ codes available at a select few research 
institutes, `HW2D` is an open-source Python reference solver that opens up the research 
field beyond these confines. As such, it significantly decreases the barrier of entry 
for fusion- and machine learning reserchers to tackle one of the biggest open questions 
in contemporary science. This implementation was and is being used for scientific papers 
on data-driven methods for plasma turbulence studies and as an introduction to plasma 
turbulence for new researchers. 


# Mathematics

The HW model describes drift-wave turbulence using two physical perturbation fields: 
the density $n$ and the potential $\phi$ using various gradients on these. 

$$
    \partial_t n = c_1 \left(\phi - n \right)
                     - \left[\phi, n \right]
                     - \kappa_n \partial_y \phi
                     - \nu \nabla^{2N} n  \\
    \partial_t \Omega = c_1 \left( \phi - n \right)
                                    - \left[ \phi, \Omega \right]
                                    - \nu \nabla^{2N} \Omega  \\
             \Omega = \nabla^2 \phi
$$

## Physical Properties

The reason why the Hasegawa-Wakatani Model has been the de-facto testing bed for new methods are its statistically stationary properties for the complex turbulent system.
The model code includes all code needed to generate these values.
It goes further, however, and provides reference values with statistical bounds for the first time for a vast range of values.
This allows simple comparison, as well es evalutaion of new methods to one reference community built resource.

$$
    \Gamma^n       = - \iint{ \mathrm{d}^2 x \space \left(n \partial_y \phi\right) } \\
    \Gamma^c       = c_1   \iint{ \mathrm{d}^2 x \space \left(n - \phi\right)^2} \\
    E              = \small \frac{1}{2} \normalsize \iint{ \mathrm{d}^2 x \space \left(n^2 - \left|\nabla_\bot \phi \right|^2 \right)} \\
    U              = \small \frac{1}{2} \normalsize \iint{ \mathrm{d}^2 x \space \left(n-\nabla_\bot^2  \phi\right)^2} = \small \frac{1}{2} \normalsize \iint{ \mathrm{d}^2 x \space \left(n-\Omega\right)^2}
$$

Additionally, spectral definitions of various properties are included, among these are:

$$
  \int{\mathrm{d} k_y \space \Gamma^n \small (k_y)}  \normalsize \space = - \int{\mathrm{d} k_y \space \left( i k_y   n\small (k_y)  \normalsize \phi^* \small (k_y)\normalsize \right) } \\
  \delta \small (k_y)  \normalsize \space = - \mathrm{Im}\left( \mathrm{log}\left( n^*\small (k_y)  \normalsize \space \phi\small (k_y)  \normalsize  \right) \right) \\
  E^N  \small (k_y)  \normalsize \space = \small \frac{1}{2}\normalsize \big| n \small (k_y) \normalsize  \big|^2 \\
  E^V  \small (k_y)  \normalsize \space = \small \frac{1}{2}\normalsize \big| k_y \phi \small (k_y) \normalsize  \big|^2 
$$

Note that it is the common practice across all reference texts to calculate $\int\cdot$ as $\langle \cdot \rangle$ in order to get comparable values for all properties.

Finally, the System does have other properties for sources and sinks that can be used to describe it, namely:

$$
    \mathfrak{D}^E = \quad \iint{ \mathrm{d}^2 x \space (n \mathfrak{D^n} - \phi \mathfrak{D}^\phi)} \\ 
    \mathfrak{D}^U = -     \iint{ \mathrm{d}^2 x \space (n - \Omega)(\mathfrak{D}^n - \mathfrak{D}^\phi)} \\
    with \quad \mathfrak{D}^n \small (x,y) \normalsize  = \nu \nabla^{2N} n \quad and \quad 
    \mathfrak{D}^\phi \small (x,y) \normalsize \space = \nu \nabla^{2N} \phi  
$$


## Reference Implementaion

The reference impementation set forth is using numerical simulations in physical space.
The methods are second order accurate in space, and fourth order accurate in time.
However, the use of the Arakawa Scheme does allow the preservation of higher order metrics.

- Gradients $\partial_x$:  2nd order accurate Central Finite Difference scheme
- Poisson Bracket $[\cdot,\cdot]$:  Arakawa Scheme [@arakawa_computational_1966]
- Spatial Derivative $\nabla_\bot$:  Laplace as 2D Central Finite Difference
- Poisson Equation $\nabla^{-2}\cdot$:  Fourier-based Poisson Solver
- Time Integration $\partial_t$:  4th order explicit Runge-Kutta scheme

