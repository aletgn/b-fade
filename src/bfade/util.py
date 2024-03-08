from typing import List, Tuple
from math import pi
import logging
import numpy as np

class MissingInputException(Exception):
    """Ensure required parameters are passed in specific contexts."""
    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(self.message)

class YieldException(Exception):
    """Ensure the precedence of particular operations/stages over others."""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

def logger_factory(name: str="root", level: str="DEBUG") -> logging.Logger:
    """
    Instantiate a logger object.

    Parameters
    ----------
    name : str
        name of the logger. The default is "root".
    level: str
        level of logging. The default is "DEBUG".

    Return
    ------
    logger : Logger from logging module.

    """
    level = level.upper()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    ch1 = logging.StreamHandler()
    ch1.setLevel(getattr(logging, level))
    logger.addHandler(ch1)

    fmt1 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            #  "%Y-%m-%d %H:%M:%S"
                            "%H:%M:%S",
                            )
    ch1.setFormatter(fmt1)

    return logger

def logger_manager(level: str, fmt: str = None) -> None:
    """
    Manage loggers. Get the loggers to modify level and format.

    Parameters
    ----------
    level : str
        level of logging. The default is "DEBUG".
    fmt : str
        format of logging

    Return
    ------
        None.
    """
    for name, logg in logging.Logger.manager.loggerDict.items():
        if isinstance(logg, logging.Logger):
            if __package__ in name:
                logg.setLevel(getattr(logging, level.upper()))
                if fmt:
                    [h.setFormatter(
                        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                        ) for h in logg.handlers]

_log = logger_factory(name=__name__, level="DEBUG")

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

def grid_factory(x1_bounds: List[float], x2_bounds: List[float], n1: int, n2: int, spacing: str = "lin") -> Tuple[np.ndarray]:
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


def sif_range(delta_sigma: np.ndarray, y: np.ndarray, sqrt_area: np.ndarray) -> np.ndarray:
    """
    Definition of Stress Intensity Factor (SIF) range, :math:`\Delta K`.

    .. math::
        \Delta K = \Delta\sigma\, Y \sqrt{\pi \sqrt{area}}

    Parameters
    ----------
    delta_sigma : array-like
        applied stress range.
    y :  array-like
        geometric factor of the defects.
    sqrt_area : array-like
         murakami's characteristic length.

    Returns
    -------
    array-like
        stress intensity factor range.

    """
    return delta_sigma * y * (pi * sqrt_area)**0.5


def inv_sif_range(delta_k: np.ndarray, delta_sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of the SIF range, thus \
        giving :math:`\sqrt{\\text{area}}`.

    .. math::
        \sqrt{\\text{area}} = {1 \over \pi} \\bigg({{\Delta K}
                                            \over {Y \Delta \sigma}}\\bigg)^2

    Parameters
    ----------
    delta_k : array-like
        stress intensity factor range.
    delta_sigma : array-like
        applied stress range.
    y : array-like
        geometric factor of the defects.

    Returns
    -------
    array-like
        sqrt_area

    """
    return ((delta_k/(y*delta_sigma))**2)/pi


def sif_equiv(sqrt_area_orig: np.ndarray, y_orig: np.ndarray, y_ref: float):
    """
    Convert sqrt_area_orig into sqrt_area according to y_ref, given y_orig\
    using the SIF-equivalence.

    .. math::
        \sqrt{\\text{area}}_{ref}=\sqrt{\\text{area}_{orig}}\,\\bigg({{Y_{orig}} \over {Y_{ref}}}\\bigg)^2

    Parameters
    ----------
    sqrt_area_orig : array-like
        original (measured) sqrt_area_data.
    y_orig : array-like
        original (indirectly retrieved from measurements) y.
    y_ref : float
        user-defined value of y set as reference.

    Returns
    -------
    array-like
        equivalent sqrt_area computed by equalling delta_k.

    """
    return ((y_orig/y_ref)**2) * sqrt_area_orig
