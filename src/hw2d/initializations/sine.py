from typing import Tuple
import numpy as np


def get_2d_sine(grid_size: Tuple[int, int], L: float) -> np.ndarray:
    """
    Generate a 2D sine pattern over a specified grid.

    This function creates a 2D array where the values are determined by the sine
    function applied to both the x and y coordinates. The resulting pattern can be
    visualized as a 2D sine wave in both the x and y directions.

    Args:
        grid_size (Tuple[int, int]): Size of the grid (height, width) for the 2D sine pattern.
        L (float): Physical extent of the domain. The resulting values will be in the range [0, L).

    Returns:
        np.ndarray: A 2D numpy array of shape `grid_size` containing the sine pattern.

    Example:
        >>> get_2d_sine((5,5), 1.0)
        array([[ 0.        ,  0.58778525,  0.95105652,  0.95105652,  0.58778525],
               [ 0.        ,  0.48163507,  0.78183148,  0.78183148,  0.48163507],
               [-0.        , -0.48163507, -0.78183148, -0.78183148, -0.48163507],
               [-0.        , -0.58778525, -0.95105652, -0.95105652, -0.58778525],
               [-0.        , -0.48163507, -0.78183148, -0.78183148, -0.48163507]])
    """
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
    return d
