from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from bfade.util import grid_factory
import scipy
from scipy.optimize import minimize_scalar

class AbstractCurve(ABC):
    
    def __init__(self, **pars):
        [setattr(self, p, pars[p]) for p in pars]
        self.pars = [k for k in pars.keys()]

    @abstractmethod
    def equation(self) -> np.ndarray:
        """
        Abstract representation of a mathematical equation.
        """
        ...

    def load_metrics(self, metrics: callable) -> None:
        """
        Load metrics to compute distances over feature plane.
        
        Parameters
        ----------
        metrics : callable
            identity, logarithm, for instance.

        Returns
        -------
        None

        """
        self.metrics = metrics

    def squared_distance(self, t: float, X: np.ndarray) -> None:
        """
        Calculate the squared distance between two points over the feature plane.
        
        Parameters
        ----------
        t : float
            Dummy parameter. Abscissa along the equation.
        X : np.ndarray
            An array representing a point belonging to the feature space [X[0], X[1]].
        
        Returns
        -------
        float
            The squared distance between the metric values of points [t, equation(t)] and X.

        """
        return (self.metrics(X[0]) - self.metrics(t))**2 +\
               (self.metrics(X[1]) - self.metrics(self.equation(t)))**2

    def squared_distance_dataset(self, t: float, X: np.ndarray) -> None:
        """
        Wraps squared_distance to compute the squared distance to each point of the datase.

        Parameters
        ----------
        t : float
            Dummy parameter. Abscissa along the equation.
        X : np.ndarray
            Dataset.

        Returns
        -------
        np.ndarray
            An array containing the squared distances between [t, equation(t)]
            and each point of the dataset.

        """
        return np.array([self.squared_distance(t, x) for x in X])
    
    def signed_distance_to_dataset(self, X):
        x_opt = []
        y_opt = []        
        l_dis = []
        dd = []
        signa = []
        
        for x in X:
            res = minimize_scalar(self.squared_distance, args=(x), 
                                  method="golden",
                                  # bounds=(0, 100000),
                                  )
            
            if res.success:
                x_opt.append(res.x)
                l_dis.append(res.fun)
            else:
                raise Exception("Error while minimising.")
        
        x_opt = np.array(x_opt)
        y_opt = self.equation(x_opt)
        
        for x, xo, yo in zip(X, x_opt, y_opt):
            d = np.array([x[0]-xo, x[1]-yo]).T
            dd.append(np.inner(d, d)**0.5)
            
            if x[1] > yo:
                signa.append(1)
            else:
                signa.append(-1)
        
        l_dis = np.array(l_dis)
        dd = np.array(dd)
        signa = np.array(signa)
        
        return dd*signa, x_opt, y_opt

    def inspect(self, x: np.ndarray, scale: str = "linear", **data: Dict) -> None:
        """
        Plot the equation of the curve and optionally the provided dataset.
        
        Parameters
        ----------
        x : np.ndarray
            Array of x-values for plotting the equation curve.
        scale : str, optional
            The scale of the plot. Default is "linear".
        **data : dict
            Additional data for scatter points. Expected keys: "X", "y".
        
        Returns
        -------
        None
        """
        fig, ax = plt.subplots(dpi=300)
        plt.plot(x, self.equation(x), "k")

        try:
            plt.scatter(data["X"][:,0], data["X"][:,1], c=data["y"], s=10)
        except:
            pass

        plt.xscale(scale)
        plt.yscale(scale)
        
    def get_curve(self) -> None:
        return self.pars, self.equation

    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"
