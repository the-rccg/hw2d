from typing import Tuple
import numpy as np
import scipy.signal


def mean_downsample2x(x: np.ndarray):
    """mean of every 2 cells, repeatedly apply to reduce higher orders

    :param x: array, first axis ignored
    :type x: array
    :return: array of reduced dimensions by (None, 2, 2, ...)
    :rtype: array
    """
    rank = len(x.shape)
    for i in range(1, rank):  # Ignore first
        if x.shape[i] == 1:  # Until first 1 dimension
            break
        a = (slice(None),) * rank  # Pass everythin
        # Now replace the "appropriate"
        x = (
            x[tuple(a[:i] + (slice(0, None, 2),) + a[i + 1 :])]
            + x[tuple(a[:i] + (slice(1, None, 2),) + a[i + 1 :])]
        ) / 2
    return x


def skip_downsampleNx(x: np.ndarray, lowerscale: float) -> np.ndarray:
    """Take every Nth value, no interpolation performed!

    :param x: input array
    :type x: array
    :param lowerscale: factor by which it is downsampled along each axis
    :type lowerscale: integer
    :return: downsampled array
    :rtype: array
    """
    # Just step through by lowerscale
    slices = [slice(None)] + [
        slice(None, None, lowerscale) for _ in range(len(x.shape) - 1)
    ]
    x = x[tuple(slices)]
    return x


def median_downsampleNx(x: np.ndarray, lowerscale: float) -> np.ndarray:
    new_shape = [x.shape[0]] + list(np.array(x.shape[1:]) // lowerscale)
    # Along axis where the shapes are not equal
    along_axis = np.arange(len(x.shape))[np.not_equal(x.shape, x.shape)]
    # Insert axis to be reduced on alter on
    sh = x.shape
    for j, i in enumerate(along_axis):
        sh = np.insert(sh, i + 1 + j, x.shape[i] // new_shape[i])
    # Group excess into new (temporary) dimensions
    x = x.reshape(sh)
    # Apply gather function along temporary dimension in reshaped array
    for ax in along_axis:
        x = np.median(x, ax + 1)
    return x


def fourier_downsample(x: np.ndarray, lowerscale: float, axes: Tuple[int] = (-1, -2)) -> np.ndarray:
    """resample in fourier space and will round output shape"""
    rounded_shape = list(map(round, np.array(x.shape) / lowerscale))
    new_lengths = np.array(rounded_shape).astype(int)
    for axis_idx in axes:
        x = scipy.signal.resample(x, num=new_lengths[axis_idx], axis=axis_idx)
    return x
5