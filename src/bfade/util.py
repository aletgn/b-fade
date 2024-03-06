import numpy as np
from typing import List


def identity(X: np.ndarray) -> None:
    """
    Return the input array unchanged.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input array.
    
    Returns
    -------
    numpy.ndarray
        Unchanged input array.
    """
    return X

def grid_factory(x1_bounds: List[float], x2_bounds: List[float], n1: int, n2: int, spacing: str = "lin") -> np.ndarray:
    """
    Create a grid of points over a 2D space.

    Parameters
    ----------
    spacing : str
        The type of spacing for the grid, either "lin" (linear) or "log" (logarithmic).
    x1 : List[float]
        A list containing the lower and upper bounds for the X-axis.
    x2 : List[float]
        A list containing the lower and upper bounds for the Y-axis.
    n1 : int
        The number of points along the X-axis.
    n2 : int
        The number of points along the Y-axis.

    Returns
    -------
    tuple
        A tuple of two arrays:
        1. X1: Flattened array of X-axis points for the entire grid.
        2. X2: Flattened array of Y-axis points for the entire grid.

    """
    if spacing == "lin":
        x1_points = np.linspace(x1_bounds[0], x1_bounds[1], n1)
        x2_points = np.linspace(x2_bounds[0], x2_bounds[1], n2)

    elif spacing == "log":
        x1_points = np.logspace(np.log10(x1_bounds[0]), np.log10(x1_bounds[1]), n1)
        x2_points = np.logspace(np.log10(x2_bounds[0]), np.log10(x2_bounds[1]), n2)

    else:
        raise KeyError("distribution spacing kind not available")

    X1, X2 = np.meshgrid(x1_points, x2_points)
    return X1.flatten(), X2.flatten()
