from typing import List, Dict, Any

import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts

from bfade.abstract import AbstractCurve
from bfade.util import grid_factory, logger_factory, YieldException, printer

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


class SyntheticDataset(Dataset):

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

    def add_noise(self, x1_std: float, x2_std: float, random_state=0) -> None:
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
        np.random.seed(0)
        if self.y is None:
            raise YieldException("Noise must be added after making classes.")
        self.X[:,0] += scipy.stats.norm(loc = 0, scale = x1_std).rvs(size=self.X.shape[0])
        self.X[:,1] += scipy.stats.norm(loc = 0, scale = x2_std).rvs(size=self.X.shape[0])

    def crop_points(self):
        _log.debug(f"{self.__class__.__name__}.{self.crop_points.__name__}")
        X = []
        y = []

        for xx, yy in zip(self.X, self.y):
            if xx[0] > 0 and xx[1] > 0:
                X.append(xx)
                y.append(yy)
            else:
                pass

        self.X = np.array(X)
        self.y = np.array(y)

