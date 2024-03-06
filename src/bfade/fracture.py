import numpy as np
from math import pi

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
    TYPE: array-like
        equivalent sqrt_area computed by equalling delta_k.

    """
    return ((y_orig/y_ref)**2) * sqrt_area_orig