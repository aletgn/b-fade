from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split as tts

from bfade.util import grid_factory, logger_factory, printer, sif_equiv, inv_sif_range, sif_range
from bfade.util import identity, printer, dummy_translator, YieldException, MissingInputException
from bfade.statistics import distribution, uniform

_log = logger_factory(name=__name__, level="DEBUG")

class Dataset:
    
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.X = None
        self.y = None
        
        try:
            self.name = kwargs.pop("name")
        except:
            self.name = "Untitled"
        
        try:
            path = kwargs.pop("path")
            reader = kwargs.pop("reader")
            self.data = reader(path, **kwargs)
        except KeyError:
            self.data = None

        try:
            self.X = self.data[["x1", "x2"]].to_numpy()
            self.y = self.data["y"].to_numpy()
            _log.debug(f"{self.__class__.__name__}.{self.__init__.__name__} -- Data ready")
        except (TypeError, KeyError):
            pass
        
        try:
            self.X = kwargs.pop("X")
            self.y = kwargs.pop("y")
            _log.debug(f"{self.__class__.__name__}.{self.__init__.__name__} -- Load data from X, y")
        except KeyError:
            pass

        try:
            self.test = kwargs.pop("test")
        except KeyError:
            self.test = None

        try:
            [setattr(self, k, kwargs[k]) for k in kwargs.keys()]
        except KeyError:
            pass       
        
        self.config()

    def config(self, save: bool = False, folder: str = "./", fmt: str = "png", dpi: int = 300) -> None:
        _log.debug(f"{self.__class__.__name__}.{self.config.__name__}")
        self.save = save
        self.folder = folder
        self.fmt = fmt
        self.dpi = dpi

    @printer
    def inspect(self, xlim=[1,1000], ylim=[1,1000], scale="linear", **kwargs):
        _log.debug(f"{self.__class__.__name__}.{self.inspect.__name__}")
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(self.X[:,0], self.X[:,1], c=self.y, s=10)

        try:
            curve = kwargs.pop("curve")
            x = kwargs.pop("x")
            ax.plot(x, curve.equation(x))
            self.name + "_curve"
        except:
            pass
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(scale)
        ax.set_yscale(scale)

        return fig, self.name + "_data"

    def partition(self, method="random", test_size = 0.2, random_state = 0):
        _log.info(f"{self.__class__.__name__}.{self.partition.__name__}")
        _log.warning(f"Train/test split. Method: {method}")
        if method == "random":
            if self.data is not None:
                data_tr, data_ts = tts(self.data,
                                    test_size=test_size,
                                    random_state=random_state)
                print(data_tr)
                return Dataset(name=self.name+"_train", **self.populate(data_tr)),\
                    Dataset(name=self.name+"_test", **self.populate(data_ts))

            elif self.X is not None and self.y is not None:
                X_tr, X_ts, y_tr, y_ts = tts(self.X, self.y,
                                            test_size=test_size,
                                            random_state=random_state)

                return Dataset(X=X_tr, y=y_tr, name=self.name+"_train"),\
                    Dataset(X=X_ts, y=y_ts, name=self.name+"_test")
            
            else:
                raise AttributeError("No data in dataset.")

        elif method == "user":
            
            if self.data is not None:
                return Dataset(name=self.name+"_train", **self.populate(self.data.query("test == 0"))),\
                Dataset(name=self.name+"_test", **self.populate(self.data.query("test == 1"))),               
            
            elif self.X is not None and self.y is not None:
                class0 = np.where(self.test == 0)
                class1 = np.where(self.test == 1)
                return Dataset(X=self.X[class0], y=self.y[class0], name=self.name+"_train"),\
                    Dataset(X=self.X[class1], y=self.y[class1], name=self.name+"_test")

            else:
                raise AttributeError("No data in dataset.")
        else:
            raise Exception("Split method incorrectly provided.")

    def populate(self, data, X_labels=["x1", "x2"], y_label="y"):
        _log.debug(f"{self.__class__.__name__}.{self.populate.__name__}")        
        return {"X": data[X_labels].to_numpy(), "y": data[y_label].to_numpy()}
    

class SyntheticData(Dataset):

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

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

    def make_tube(self, curve, x_bounds: List[float], n: int = 50, up: float = 0.1,
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
            x1 = np.linspace(x_bounds[0], x_bounds[1], n)
        
        else:
            steps = np.logspace(up, down, step)
            x1 = np.logspace(np.log10(x_bounds[0]), np.log10(x_bounds[1]), n)

        x2 = curve.equation(x1)
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

    def make_classes(self, curve):
        """
        Assign class labels to the synthetic dataset based on the underlying curve.
     
        Returns
        -------
        None
     
        """   
        _log.debug(f"{self.__class__.__name__}.{self.make_classes.__name__}")    
        self.y = []
        for d in self.X:
            if curve.equation(d[0]) > d[1]:
                self.y.append(0)
            else:
                self.y.append(1)
        self.y = np.array(self.y)

    def clear_points(self, curve, tol: float = 1e-2):
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
            self.X = np.array([d for d in self.X if abs(curve.equation(d[0]) - d[1]) > tol])

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
        if self.y is None:
            raise YieldException("Noise must be added after making classes.")
        self.X[:,0] += scipy.stats.norm(loc = 0, scale = x1_std).rvs(size=self.X.shape[0])
        self.X[:,1] += scipy.stats.norm(loc = 0, scale = x2_std).rvs(size=self.X.shape[0])


class ElHaddadData(Dataset):

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def pre_process(self, **kwargs):
        """
        Pre-process the dataset:

             - set 'Y'

             - convert sqrt_area using the SIF equivalence

             - compute SIF

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
        _log.debug(f"{self.__class__.__name__}.{self.pre_process.__name__}")
        try:
            self.Y = kwargs.pop("Y_ref")
            _log.warning(f"Y_ref user-provided = {self.Y:.2f}")
        except KeyError:
            _log.warning(f"Y_ref not user-provided")
            _log.warning("Verify uniqueness of Y")
            if len(set(self.data.Y)) == 1:
                self.Y = list(set(self.data.Y))[0]
                _log.warning(f"Y is unique = {self.Y:.2f}")              
            else:
                _log.error(f"Y is not unique")
                _log.debug(f"Values found: {set(self.data.Y)}")
                raise MissingInputException("Y_ref is neither unique nor provided")

        _log.info("Update dataframe")
        self.data.rename(columns={"Y": "Y_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("Y_")+1, "Y", self.Y)

        _log.warning(f"Convert sqrt_area by {self.Y:.2f}")
        self.data.rename(columns={"sqrt_area": "sqrt_area_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("sqrt_area_")+1, "sqrt_area",
                        sif_equiv(self.data.sqrt_area_, self.data.Y_, self.Y))

        _log.info("Compute SIF range")
        self.data.insert(list(self.data.columns).index("Y")+1, "dk",
                        sif_range(self.data.delta_sigma, self.data.Y, self.data.sqrt_area*1e-6))

        _log.debug(f"Calculate min max of delta_k for colour bars")
        self.aux = self.data["dk"].to_numpy()
        self.aux_min = self.aux.min()
        self.aux_max = self.aux.max()

        self.X = self.data[["sqrt_area", "delta_sigma"]].to_numpy()
        self.y = self.data["failed"].to_numpy()
        self.Y = self.data["Y"].to_numpy()
        self.aux = self.data["dk"].to_numpy()
        self.aux_min = self.aux_min
        self.aux_max = self.aux_max

    def populate(self, data, X_labels=["sqrt_area", "delta_sigma"], y_label="failed"):
        return super().populate(data, X_labels, y_label)