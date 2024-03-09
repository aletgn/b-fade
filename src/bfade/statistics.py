from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.stats import t as t_student
from bfade.util import MissingInputException, logger_factory

_log = logger_factory(name=__name__, level="DEBUG")

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


class MonteCarlo:

    def __init__(self, n_samples: int, curve, confidence: float = 95) -> None:
        """
        Initialise Monte Carlo simulation

        Parameters
        ----------
        samples : int
            number of samples to draw.
        confidence : int, optional
            Confidence level of the prediction interval. The default is 95.
        curve : AbstractCurve
            Reference curve.    

        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.__init__.__name__}")
        self.n_samples = n_samples
        self.curve = curve
        self.confidence = confidence

    def sample_joint(self, bayes) -> None:
        """
        Sample the joint posterior distribution.

        Parameters
        ----------
        bayes : AbstractBayes

        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.sample_joint.__name__}")
        self.pars = bayes.pars
        self.samples = bayes.joint.rvs(self.n_samples)

    def sample_marginals(self, bayes) -> None:
        """
        Sample the marginal posterior distributions.

        Parameters
        ----------
        bayes : AbstractBayes


        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.sample_marginals.__name__}")
        self.pars = bayes.pars
        self.samples = np.array([getattr(bayes, "marginal_" + p).rvs(self.n_samples) for p in bayes.pars]).T

    def prediction_interval(self, x_edges: List[float], n: int, spacing: str, **args: Dict[str, Any]) -> Tuple:
        """
        Compute prediction intervals for a curve.

        .. math::
            P\\Big[\overline{\mathcal{E}^{(\sf M)}} \
                            - \mathcal{P}^{(\sf M)} \
                                \le \mathcal{E}^{({\sf M}+1)} \
                                    \le \overline{\mathcal{E}^{(\sf M)}} \
                                        + \mathcal{P}^{(\sf M)}\\Big] = \\beta

        where :math:`\\beta` is the confidence level. The semi ampliture of \
            the prediction interval is:

        .. math::
            \mathcal{P}^{(\sf M)} = T_{\\beta} S^{(\sf M)}  \sqrt{1 + 1/{\mathsf{M}}}

        Parameters
        ----------
        spacing : str, optional
            spacing for x and y axes.
        x_edges : list of float, optional
            Edges of the x-axis over which the curve is plotted.
        n : int, optional
            Resolution of the curve (number of points over x-axis). The default is 100.
        kwargs:
            extra input parameters of the curve not included in AbstractBayes
        Returns
        -------
        result : Tuple
            A tuple containing the following elements:

            - 'mean': The expected curve data.

            - 'pred': The semi-amplitude of the prediction interval.

            - 'x1': abscissa along with the prediction interval is computed.

        """
        _log.info(f"{self.__class__.__name__}.{self.prediction_interval.__name__}")
        curves = []
        if spacing == "log":
            x1 = np.logspace(np.log10(x_edges[0]), np.log10(x_edges[1]), n)
        else:
            x1 = np.linspace(x_edges[0], x_edges[1], n)

        for s in self.samples:
            d = dict(zip(self.pars, s))
            d.update(**args)
            curves.append(self.curve(**d).equation(x1))

        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        pred = t_student(df=len(curves)-1).ppf(self.confidence/100)*std*((1+1/len(curves))**0.5)

        return mean, pred, x1