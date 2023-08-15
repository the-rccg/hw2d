"""
`fourier_noise.py`: A module to generate 2D noise patterns using the Fast Fourier Transform (FFT).

This module provides functionality to generate 2D noise patterns based on Fourier space manipulation.
The generated noise can be controlled by specifying the desired resolution, size, scale, frequency range, 
and wavelength range. The resulting noise can be useful for various applications in simulations, testing,
and visualization.
"""
import numpy as np
from typing import Tuple


def get_fft_noise(
    resolution: Tuple[int, int],
    size: float,
    scale: float,
    min_frequency: float = 0,
    max_frequency: float = 0,
    min_wavelength: float = 0,
    max_wavelength: float = 0,
    factor: int = 2,
) -> np.ndarray:
    """
    Generate a 2D noise pattern using the FFT.

    This function creates a 2D noise pattern based on standard normal distributions in Fourier space.
    It allows for control over the frequency and wavelength components of the noise through the provided parameters.

    Args:
        resolution (Tuple[int, int]): The dimensions of the generated noise.
        size (float): Physical size.
        scale (float): Scaling factor for the frequencies.
        min_frequency (float, optional): Minimum frequency for filtering. Default is 0.
        max_frequency (float, optional): Maximum frequency for filtering. Default is 0.
        min_wavelength (float, optional): Minimum wavelength for filtering. Default is 0.
        max_wavelength (float, optional): Maximum wavelength for filtering. Default is 0.
        factor (int, optional): Factor used in calculating the frequency components. Default is 2.

    Returns:
        np.ndarray: The generated 2D noise pattern.
    """
    # Calculate random complex values
    shape = (1, *resolution, 1)
    rnd_real = np.random.standard_normal(shape).astype(np.complex128)
    rnd_imag = 1j * np.random.standard_normal(shape).astype(np.complex128)
    rndj = rnd_real + rnd_imag

    # Calculate frequency components
    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution], indexing="ij")
    k = np.expand_dims(np.stack(k, -1), 0)
    k = k * resolution / size * scale  # in physical units
    k = np.sum(np.abs(k) ** factor, axis=-1, keepdims=True)

    # Convert wavelengths to frequencies if provided
    if max_wavelength:
        min_frequency = 1 / max_wavelength
    if min_wavelength:
        max_frequency = 1 / min_wavelength

    # Create frequency mask
    weight_mask = np.ones(shape)
    if min_frequency:
        weight_mask += 1 / (1 + np.exp((min_frequency - k) * 1e3)) - 1
    if max_frequency:
        with np.errstate(divide="ignore", invalid="ignore"):
            weight_mask -= 1 / (1 + np.exp((max_frequency - k) * 1e3))
    # Check weight mask
    assert np.all(weight_mask <= 1) and np.all(
        weight_mask >= 0
    ), "Weight mask values out of bounds."

    # Handle division by zero for k
    k[(0,) * len(k.shape)] = np.inf
    inv_k = 1 / k
    inv_k[(0,) * len(k.shape)] = 0

    # Compute result
    smoothness = 1
    fft = rndj * inv_k**smoothness * weight_mask
    array = np.real(np.fft.ifft2(fft, axes=[1, 2]))
    array /= np.std(array, axis=tuple(range(1, len(array.shape))), keepdims=True)
    array -= np.mean(array, axis=tuple(range(1, len(array.shape))), keepdims=True)
    array = array.astype(np.float64)
    return array[0, ..., 0]
