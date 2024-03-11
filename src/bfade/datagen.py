from typing import List, Dict, Any

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bfade.abstract import AbstractCurve
from bfade.util import grid_factory, logger_factory, YieldException, printer

_log = logger_factory(name=__name__, level="DEBUG")

class SyntheticDataset:
    """
    A class representing a synthetic dataset generated from a given curve.

    """
    def __init__(self, curve: AbstractCurve, **kwargs: Dict[str, Any]) -> None:
        """
        

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Optional keyword arguments, such as 'name'.

        curve : AbstractCurve
            An instance of the AbstractCurve class representing the underlying curve.

        Returns
        -------
        None

        """
        try:
            self.name = kwargs.pop("name")
        except KeyError:
            self.name = "Untitled"

        self.pars, self.equation = curve.get_curve()
        self.X = None
        self.y = None
        _log.debug(f"{self.__class__.__name__}.{self.__init__.__name__} -- {self}")
        self.config()

    def config(self, save: bool = False, folder: str = "./", fmt: str = "png", dpi: int = 300) -> None:
        _log.debug(f"{self.__class__.__name__}.{self.config.__name__}")
        self.save = save
        self.folder = folder
        self.fmt = fmt
        self.dpi = dpi
        
    def make_grid(self, x1_bounds: List[float], x2_bounds: List[float],
                  n1: int, n2: int, spacing: str ="lin") -> None:
        """
        Generate a grid of input points for the synthetic dataset.
        
        Parameters
        ----------
        x1_bounds : List[float]
            Bounds for the first feature (x1).
        x2_bounds : List[float]
            Bounds for the second feature (x2).
        n1 : int
            Number of points along the first dimension (x1).
        n2 : int
            Number of points along the second dimension (x2).
        scale : str, optional
            The scale of the grid spacing, either "lin" for linear or "log" for logarithmic.
            Default is "lin".
        
        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.make_grid.__name__}")
        self.X = np.vstack(grid_factory(x1_bounds, x2_bounds, n1, n2, spacing)).T
    
    def make_tube(self, x1_bounds: List[float], n: int = 50, up: float = 0.1,
                  down: float = -0.1, step: int = 4, spacing: str = "lin") -> None:
        """
        Generate a ``tube'' of points surrounding the given EH curve.
        
        This method should be used in place of make_grid.

        The dataset is inspected via view_grid

        Parameters
        ----------
        xlim : List[float]
            Edges of the interval along the x-axis.
        x_res : int, optional
            Number of points . The default is 50.
        up : float, optional
            Maximum upward translation of the EH curve. The default is 0.1.
            Note that in log-space (uniform) translations is achieved via
            multiplication.
        down : float, optional
            Minimum downward translation of the EH curve. The default is -0.1.
            Note that in log-space (uniform) translations is achieved via
            multiplication.
        step : int, optional
            Number of translated curves. The default is 12. The method disregards
            the curve obtained via translation when the multiplication factor
            is 1. It gives the original curve, where points are classified as
            0.5, so they do not bring about any information.
        spacing: str, optional
            Spacing of the points.

        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.make_tube.__name__}")
        assert down < up
        if spacing == "lin":
            steps = np.linspace(up, down, step)
            x1 = np.linspace(x1_bounds[0], x1_bounds[1], n)
        
        else:
            steps = np.logspace(up, down, step)
            x1 = np.logspace(np.log10(x1_bounds[0]), np.log10(x1_bounds[1]), n)

        x2 = self.equation(x1)
        X1 = []
        X2 = []
        for s in steps:
            if spacing == "lin":
                X2.append(x2 + s)
            else:
                X2.append(x2 * s)
        X2 = np.array(X2)
        X1 = np.array(list(x1)*X2.shape[0]).flatten()
        X2 = X2.flatten()
        self.X = np.vstack([X1,X2]).T
    
    def clear_points(self, tol: float = 1e-2):
        """
        Remove data points from the synthetic dataset based on the deviation from the underlying curve.
    
        Parameters
        ----------
        tol : float, optional
            Tolerance level for determining the deviation. Points with a deviation less than `tol` will be removed.
            Default is 1e-2.
    
        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.clear_points.__name__} -- tol = {tol}")
        if self.y is not None:
            raise YieldException("Points must cleared before making classes.")
        else:
            self.X = np.array([d for d in self.X if abs(self.equation(d[0]) - d[1]) > tol])
        
    def make_classes(self):
        """
        Assign class labels to the synthetic dataset based on the underlying curve.
     
        Returns
        -------
        None
     
        """   
        _log.debug(f"{self.__class__.__name__}.{self.make_classes.__name__}")    
        self.y = []
        for d in self.X:
            if self.equation(d[0]) > d[1]:
                self.y.append(0)
            else:
                self.y.append(1)
        self.y = np.array(self.y)
    
    def add_noise(self, x1_std: float, x2_std: float) -> None:
        """
        Add Gaussian noise to the data points in the synthetic dataset.
    
        Parameters
        ----------
        x1_std : float
            Standard deviation of the Gaussian noise to be added to the first feature (x1).
        x2_std : float
            Standard deviation of the Gaussian noise to be added to the second feature (x2).
    
        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.add_noise.__name__}")
        self.X[:,0] += scipy.stats.norm(loc = 0, scale = x1_std).rvs(size=self.X.shape[0])
        self.X[:,1] += scipy.stats.norm(loc = 0, scale = x2_std).rvs(size=self.X.shape[0])
    
    @printer
    def inspect(self, x: np.ndarray = None, scale="linear"):
        """
        Visualize the synthetic dataset and optionally plot the underlying curve.
    
        Parameters
        ----------
        x : np.ndarray, optional
            Input values for evaluating the underlying curve. If provided, the curve will be plotted.
        scale : str, optional
            Scale for both x and y axes. Options are "linear" (default) or "log".
    
        Returns
        -------
        None
            Displays a scatter plot of the synthetic dataset with an optional plot of the underlying curve.
        """
        _log.debug(f"{self.__class__.__name__}.{self.inspect.__name__}")
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(self.X[:,0], self.X[:,1], c=self.y, s=10)
        
        try:
            plt.plot(x, self.equation(x), "k")
        except:
            pass
        
        plt.xscale(scale)
        plt.yscale(scale)
        plt.show()
        
        return fig, self.name + "_dataset"
        
    def __repr__(self) -> str:
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"
