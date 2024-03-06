from typing import Dict, Any
import numpy as np
from bfade.util import MissingInputException

class distribution():
    """Interface to scipy random variables.
    
    This class istantiates distribution probilities relying on scipy.stats.
    """

    def __init__(self, dist, **kwargs: Dict[str, Any]) -> None:
        """
        Initialize a distribution.

        Parameters
        ----------
        dist : callable
            Function representing a probability distribution
            (e.g., scipy.stats.norm for a normal distribution).
        **kwargs : Dict[str, Any]
            Parameters of the selected distribution.

        Returns
        -------
        None.

        """
        self.dist_pars = kwargs
        self.dist = dist(**kwargs)

    def pdf(self, x: np.ndarray) -> float:
        """
        Probability density function (PDF).

        Parameters
        ----------
        x : float
            The point where PDF is evaluated.

        Returns
        -------
        float
            The PDF value at `x`.

        """
        return self.dist.pdf(x)
    
    def logpdf(self, x: np.ndarray) -> float:
        """
        Log-probability density function (PDF).

        Parameters
        ----------
        x : float
            The point where PDF is evaluated.

        Returns
        -------
        float
            The Log-PDF value at `x`.

        """
        return self.dist.logpdf(x)
    
    def rvs(self, size: int) -> np.ndarray:
        """
        Sample PDF.

        Parameters
        ----------
        size : int
            Size of the sample.
            
        Returns
        -------
        np.ndarray
            Sample of size 'size'

        """
        return self.dist.rvs(size)
    
    def __str__(self) -> str:
        _dict = self.__dict__.pop("dist_pars")
        return f"\n{_dict}"


class uniform():
    """Uniform probability distribution.

    This class is provided for convenience. Using this class is totally optional
    and one can alternatively utilise scipy's either. Still scipy's uniform
    returns 0 if the input point is outside the set range. This can cause issues
    to the likelihood for under-conservative choices of the lower and the upper
    bound of the range. Therefore, this custom uniform distribution is designed
    to return a given value everywhere. Obviosly, this constrast the CDF but
    facilitates the computation of the likelihood.

    The methods simulate part of the typical interface of scipy's random vars.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize Uniform distribution.

        Parameters
        ----------
        unif_value : float
            The uniform value associated with the distribution.

        Raises
        ------
        MissingInputException
            If `unif_value` is not provided.

        """
        # _log.info(f"(Non-scipy). {self.__class__.__name__}.{self.__init__.__name__}")
        try:
            self.unif_value = kwargs.pop("unif_value")
        except KeyError as KE:
            raise MissingInputException(f"{KE} not provided")

        self.dist = self.unif_value

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function (PDF) at given x."""
        return np.ones(np.array(x).shape)*self.unif_value
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function (PDF) at given x."""
        return np.log(np.ones(np.array(x).shape)*self.unif_value)

    def rvs(self, size: int) -> np.ndarray:
        """Draw a random sample of size 'size'"""
        return np.ones(size)*self.unif_value