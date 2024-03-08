from typing import List, Dict, Any

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bfade.curve import AbstractCurve
from bfade.util import grid_factory, MissingInputException, YieldException
from bfade.fracture import sif_equiv, sif_range

from sklearn.model_selection import train_test_split as tts

import pandas as pd

from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    
    def __init__(self, **kwargs):
        
        [setattr(self, k, kwargs[k]) for k in kwargs.keys()]
        # self.train = None
        # self.test = None
        
        # self.X = None
        # self.y = None
        
    def read(self, reader: callable, filename: str, folder: str = "./", **kwargs: Dict[str, Any]) -> None:
        self.data = reader(folder + filename)
    
    def enter(self, X, y, test, **kwargs):
        self.X = X
        self.y = y
        self.test = test
        [setattr(self, k, kwargs[k]) for k in kwargs]
        
    @abstractmethod
    def pre_process():
        pass
    
    @abstractmethod
    def populate():
        pass
    
    def partition():
        pass
    
class ElHaddadDataset(AbstractDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def pre_process(self, **kwargs):
        """
        Pre-process the dataset:

             - set 'Y'

             - convert sqrt_area using the SIF equivalence

             - compute SIF dk

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Y_ref to specify the reference value for Y.

        Raises
        ------
        MissingInputException
            Raised if 'Y' is neither unique in the dataset nor provided\
            as a keyword argument.

        """
        # _log.debug(f"{self.__class__.__name__}.{self.pre_process.__name__}")
        try:
            self.y_ref = kwargs.pop("Y_ref")
            # _log.warning(f"y_ref user-provided = {self.y_ref:.2f}")
        except KeyError:
            # _log.warning(f"y_ref not user-provided")
            # _log.warning("Verify uniqueness of y")
            if len(set(self.data.Y)) == 1:
                self.Y = list(set(self.data.Y))[0]
                # _log.debug(f"y_ref is unique = {self.y_ref:.2f}")              
            else:
                # _log.error(f"y is not unique")
                # _log.debug(f"Values found: {set(self.data.y)}")
                raise MissingInputException("y_ref is neither unique nor provided")

        # _log.info("Update dataframe")
        self.data.rename(columns={"Y": "Y_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("Y_")+1, "Y", self.Y)

        # _log.warning(f"Convert sqrt_area by {self.y_ref:.2f}")
        self.data.rename(columns={"sqrt_area": "sqrt_area_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("sqrt_area_")+1, "sqrt_area",
                        sif_equiv(self.data.sqrt_area_, self.data.Y_, self.Y))

        # _log.info("Compute SIF range")
        self.data.insert(list(self.data.columns).index("Y")+1, "dk",
                        sif_range(self.data.delta_sigma, self.data.Y, self.data.sqrt_area*1e-6))

        # _log.debug(f"Calculate min max of delta_k for colour bars")
        self.aux_min = self.data.dk.min()
        self.aux_max = self.data.dk.max()
        
        return ElHaddadDataset(**self.populate("data"))
        
    def partition(self, method: str = "random", test_size: float = 0.2, rnd_state: int = 0) -> None:
        """
        Split dataset into seen (training) and unseen (test) points.

        Parameters
        ----------
        method : string, optional
            the parameters controls how to partition (split) the dataset. \
                if "random", then use the built in function of \
                    sklearn train_test_split. Else, if "user", then split\
                        according to the column "split" in the dataset.
            The default is "random".
        test_size : float, optional
            test_size controls the fraction of unseen (test) data against\
                those considered for MAP. The default is 0.2. \
                    Accepted values from 0.0.
            to 1.0
        rnd_state : int, optional
            random state for splitting. The default is 0.

        Raises
        ------
        Exception
            if "method" is not included in the possible choices, then throw an
            exception.

        Returns
        -------
        None.

        """
        # _log.debug(f"{self.__class__.__name__}.{self.partition.__name__}")
        # if self.scaler_delta_k or self.scaler_delta_sigma or self.scaler_sqrt_area:
            # raise YieldException("Partitioning must be done before defining scalers")
        # _log.warning(f"Train/test split. Method: {method}")
        if method == "random":
            self.split_method = method
            self.train, self.test = tts(self.data, test_size=test_size,
                                    random_state=rnd_state, shuffle=True)
        elif method == "user":
            self.split_method = method
            self.train = self.data.query("test == 0")
            self.test = self.data.query("test == 1")
        else:
            raise Exception("split method incorrectly provided")
            
        return ElHaddadDataset(**self.populate("train")), ElHaddadDataset(**self.populate("test"))
            
    def populate(self, data):
        return {"X": getattr(self, data)[["sqrt_area", "delta_sigma"]].to_numpy(),
                "y": getattr(self, data)["failed"].to_numpy(),
                # "aux": getattr(self, data)["dk"].to_numpy(),
                # "aux_min": self.aux_min,
                # "aux_max": self.aux_max,
                "Y": self.Y}
        
class SyntheticDataset:
    """
    A class representing a synthetic dataset generated from a given curve.

    """
    def __init__(self, curve: AbstractCurve) -> None:
        """
        

        Parameters
        ----------
        curve : AbstractCurve
            An instance of the AbstractCurve class representing the underlying curve.

        Returns
        -------
        None

        """
        self.pars, self.equation = curve.get_curve()
        self.X = None
        self.y = None
        
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
        self.X = np.array([d for d in self.X if abs(self.equation(d[0]) - d[1]) > tol])
        
    def make_classes(self):
        """
        Assign class labels to the synthetic dataset based on the underlying curve.
     
        Returns
        -------
        None
     
        """       
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
        self.X[:,0] += scipy.stats.norm(loc = 0, scale = x1_std).rvs(size=self.X.shape[0])
        self.X[:,1] += scipy.stats.norm(loc = 0, scale = x2_std).rvs(size=self.X.shape[0])
    
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
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(self.X[:,0], self.X[:,1], c=self.y, s=10)
        
        try:
            plt.plot(x, self.equation(x), "k")
        except:
            pass
        
        plt.xscale(scale)
        plt.yscale(scale)
