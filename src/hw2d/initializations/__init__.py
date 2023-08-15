"""
`hw2d.initializations`: Initialization methods for 2D patterns in the HW2D system.

This module provides a collection of functions to generate initial patterns or noise
for the HW2D system. The methods encompass both deterministic patterns (like sine waves)
and stochastic patterns (like Fourier noise). These initializations can be useful for
starting simulations, testing, and visualization in the context of the HW2D system.

Functions:
    - `get_2d_sine`: Generates a 2D sine wave pattern over a specified grid.
    - `get_fft_noise`: Creates a 2D noise pattern using the Fast Fourier Transform.

Example Usage:
    >>> # Generate a 2D sine pattern
    >>> sine_pattern = get_2d_sine((256, 256), 1.0)
    
    >>> # Generate a 2D noise pattern using FFT
    >>> noise_pattern = get_fft_noise((256, 256), 1.0, 1.0, min_frequency=0.1, max_frequency=2.0)
"""
