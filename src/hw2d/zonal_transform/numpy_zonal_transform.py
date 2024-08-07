import numpy as np


def zonal_transform_fourier(field: np.ndarray) -> np.ndarray:
    # Transform to (..., k_y, x)
    field_ky = np.fft.fft(field, norm="ortho", axes=(-2))
    # Average over ky
    avg_ky_field = np.sum(field_ky, axes=(-2), keep_dims=True)
    # Rescale by box size
    # avg_ky_field /= self.L
    # Reshape original field
    non_zonal_field = field - avg_ky_field
    return non_zonal_field


def zonal_transform(field: np.ndarray) -> np.ndarray:
    # Average over y
    avg_ky_field = np.mean(field, axis=(-2), keepdims=True)
    # Reshape original field
    non_zonal_field = field - avg_ky_field
    return non_zonal_field
