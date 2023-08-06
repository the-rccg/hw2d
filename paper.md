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

Nuclear fusion provides the promise of clean, endless energy for humanity
in the future. One of the biggest open questions standing in the way of 
commercially viable fusion devices is containing the energy within plasma that 
is transported out through drift wave turbulence. However, the immense complexity 
of modeling turbulence inside potential reactors still remains outside the realm 
of the biggest supercomputers. Hence, the development of new methods for simulating 
plasma turbulence is a pressing issue. The community pushes for the introduction 
of new numerical- and machine learning methods - most commonly tested for 
viability with the Hasegawa-Wakatani model [@hasegawa_wakatani;@Yatomi_2023;@grillix;@anderson2020elucidating;@goumiri2013reduced;@heinonen2020learning].
Up to now, however, there is no open-source reference implementation to verify 
against or build upon.


# Statement of need

`HW2D` provides an easy to understand, easy to use, and easy to extend 
open-source implementation to help accelerate the testing of new ideas and methods.
As such, it significantly decreases the barrier of entry for fusion- and machine 
learning reserchers to tackle one of the biggest open questions in contemporary science.
In contrast to closed-source Fortran and C++ codes, `HW2D` is an open-source Python 
reference solver opens the field beyond a select number of research institutions.
This implementation was and is being used for scientific inquiry into data-driven methods
for plasma turbulence studies and as an introduction to plasma turbulence for 
new researchers. 

`HW2D` serves as one central reference against which methods can be benchmarked, 
validated, and trained with a clear set of reference implementations for comparison. 
The similarity of HW to Navier-Stokes (NS) [@camargo], makes it a prime candidate for 
transfering insights gathered from methods developed for NS fluids towards fusion 
turbulence theory. Considering the turbulent and chaotic nature of plasma turbulence, 
verifying implementations can be resource intensive --- this is now automated with 
CI pipelines and open-sourced. 


# Mathematics

The HW model describes drift-wave turbulence using two physical fields: the density $n$ and the potential $\phi$ using various gradients on these. 

$$
\begin{align}
    \partial_t n &= c_1 \left( \phi - n \right)
                     - \left[ \phi, n \right]
                     - \kappa_n \partial_y \phi
                     - \nu \nabla^{2N} n  \\
    \partial_t \Omega &= c_1 \left( \phi - n \right)
                                    - \left[ \phi, \Omega \right]
                                    - \nu \nabla^{2N} \Omega  \\
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
    \Gamma^n       &= - \iint{ \mathrm{d}^2 x \space \left(n \partial_y \phi\right) } \\
    \Gamma^c       &= c_1   \iint{ \mathrm{d}^2 x \space \left(n - \phi\right)^2} \\
    E              &= \small \frac{1}{2} \normalsize \iint{ \mathrm{d}^2 x \space \left(n^2 - \left|\nabla_\bot \phi \right|^2 \right)} \\
    U              &= \small \frac{1}{2} \normalsize \iint{ \mathrm{d}^2 x \space \left(n-\nabla_\bot^2  \phi\right)^2} = \small \frac{1}{2} \normalsize \iint{ \mathrm{d}^2 x \space \left(n-\Omega\right)^2}
\end{align}
$$

Additionally, spectral definitions of various properties are included, among these are:

$$
\begin{align}
  \int{\mathrm{d} k_y \space \Gamma^n \small (k_y)}  \normalsize \space &= - \int{\mathrm{d} k_y \space \left( i k_y   n\small (k_y)  \normalsize \phi^* \small (k_y)\normalsize \right) } \\
  \delta \small (k_y)  \normalsize \space &= - \mathrm{Im}\left( \mathrm{log}\left( n^*\small (k_y)  \normalsize \space \phi\small (k_y)  \normalsize  \right) \right) \\
  E^N  \small (k_y)  \normalsize \space &= \small \frac{1}{2}\normalsize \big| n \small (k_y) \normalsize  \big|^2 \\
  E^V  \small (k_y)  \normalsize \space &= \small \frac{1}{2}\normalsize \big| k_y \phi \small (k_y) \normalsize  \big|^2 
\end{align}
$$

Note that it is the common practice across all reference texts to calculate $\int\cdot$ as $\langle \cdot \rangle$ in order to get comparable values for all properties.

Finally, the System does have other properties for sources and sinks that can be used to describe it, namely:

$$
\begin{align}
    \mathfrak{D}^E &= \quad \iint{ \mathrm{d}^2 x \space (n \mathfrak{D^n} - \phi \mathfrak{D}^\phi)} \\ 
    \mathfrak{D}^U &= -     \iint{ \mathrm{d}^2 x \space (n - \Omega)(\mathfrak{D}^n - \mathfrak{D}^\phi)} \\
    with \quad \mathfrak{D}^n \small (x,y) \normalsize  &= \nu \nabla^{2N} n \quad and \quad 
    \mathfrak{D}^\phi \small (x,y) \normalsize \space = \nu \nabla^{2N} \phi  
\end{align}
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


# Figures



# Acknowledgements

...

# References
