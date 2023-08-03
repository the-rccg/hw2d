---
title: 'HW2D: A reference implementation of the Hasegawa-Wakatani model for plasma turbulence in Fusion reactors'
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
in the future. However, the immense complexity of modeling turbulence inside
potential reactors still remains outside the realm of the biggest supercomputers.
Hence, the development of new methods for simulating plasma turbulence is 
a pressing issue. The community pushes to introduce new numerical and machine
learning methods that are tested versus the Hasegawa-Wakatani model [@hasegawa_wakatani;@Yatomi_2023;@grillix;@anderson2020elucidating;@goumiri2013reduced;@heinonen2020learning].
Up to now, however, there is no reference implementation to verify or build upon.
This package is providing an easy to understand, easy to use, and easy to extend 
reference implementation to test new ideas and methods without having to 
waste resources to find stable parameters.
As such, it significantly decreases the barrier of entry for fusion- and machine 
learning reserachers to tackle one of the biggest open questions in contemporary science.

# Statement of need

`HW2D` was designed with easy of understanding front and center. It is the reference 
implementation that was and is being used for scientific inquiry into data-driven methods
for plasma turbulence studies, as well as introducing researchers to plasma turbulence.
It provides one central reference against which methods can be benchmarked, validated, 
and trained with a clear set of reference values for comparison.
The reference solver furthemore allows intrusive machine learning models to be developed 
without having to verify a new solver. 
The similarity of HW to Navier-Stokes, makes it a prime candidate for transfering insights 
gathered from methods developed for NS fluids towards fusion turbulence theory [@camargo].
Considering the turbulent, chaotic nature of plasma turbulence, verifying implementations 
can be resource intensive --- and is now automated with CI pipelines.
Keeping a simple structure for the project allows expansion by all levels of programming 
knowledge:bachelor students to professional machine learning researchers.


# Mathematics

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
\Gamma^n\scriptstyle(k_y)\displaystyle\; = -\!\! \int{\!\mathrm{d} k_y \;\, i k_y \, n\scriptstyle(k_y) \, \displaystyle\phi\scriptstyle(k_y)\displaystyle^* }
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
