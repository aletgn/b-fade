from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize_scalar, minimize

from bfade.util import grid_factory
from bfade.util import identity
from bfade.statistics import distribution, uniform

class AbstractCurve(ABC):
    
    def __init__(self, metrics: callable = identity, **pars) -> None:
        """
        Initialise curve.
        
        Parameters
        ----------
        metrics : callable
            identity, logarithm, for instance. Default is identity.
            
        pars: Dict
            parameters of the curve.
        

        Returns
        -------
        None

        """
        try:
            self.name = pars.pop("name")
        except KeyError:
            self.name = "Untitled"
            
        try:
            self.metrics = pars.pop("metrics")
        except KeyError:
            self.metrics = identity
        
        [setattr(self, p, pars[p]) for p in pars]
        self.pars = [k for k in pars.keys()]
        self.metrics = metrics

    @abstractmethod
    def equation(self) -> np.ndarray:
        """
        Abstract representation of a mathematical equation.
        """
        ...

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

    # def squared_distance_dataset(self, t: float, X: np.ndarray) -> None:
    #     """
    #     Wraps squared_distance to compute the squared distance to each point of the dataset.

    #     Parameters
    #     ----------
    #     t : float
    #         Dummy parameter. Abscissa along the equation.
    #     X : np.ndarray
    #         Dataset.

    #     Returns
    #     -------
    #     np.ndarray
    #         An array containing the squared distances between [t, equation(t)]
    #         and each point of the dataset.

    #     """
    #     return np.array([self.squared_distance(t, x) for x in X])
    
    def signed_distance_to_dataset(self, X):
        """
        Wraps squared_distance to compute the squared distance to each point of the dataset.
        """
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
            # else:
            #     raise Exception("Error while minimising.")
        
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
        data : dict
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
        plt.show()
        
    def inspect_signed_distance(self, x: np.ndarray, x_opt: np.ndarray, y_opt: np.ndarray, dis: np.ndarray,
                                X: np.ndarray = None, y: np.ndarray = None, scale: str = "linear"):
        """
        Visualize the signed distance of data points to an minumum-distance (optimal) point
        along the curve.
    
        Parameters
        ----------
        x : np.ndarray
            Input values for the optimal point.
        x_opt : np.ndarray
            x-coordinate of the optimal point.
        y_opt : np.ndarray
            y-coordinate of the optimal point.
        dis : np.ndarray
            Signed distance values for each data point.
        X : np.ndarray, optional
            Input features of the synthetic dataset.
        y : np.ndarray, optional
            Target values of the synthetic dataset.
        scale : str, optional
            Scale for both x and y axes. Options are "linear" (default) or "log".
    
        Returns
        -------
        None
            Displays a plot visualizing the signed distance of data points to the optimal point.

        """
        fig, ax = plt.subplots(dpi=300)
        
        if X is not None:
            plt.scatter(X[:,0], X[:,1], c=y)
        
        plt.scatter(x_opt, y_opt)
        
        for x, xo, yo, d in zip(X, x_opt, y_opt, dis):
            if d > 0:
                ax.plot([x[0], xo], [x[1], yo], '-b')
            else:
                ax.plot([x[0], xo], [x[1], yo], '-.r')
                
        ax.axis("equal")
        plt.xscale(scale)
        plt.yscale(scale)
        plt.show()
        
    def get_curve(self) -> None:
        return self.pars, self.equation

    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"


class AbstractBayes(ABC):
    """Bayesian framework to perform Maximum a Posterior Estimation and predictions."""
    
    def __init__(self, *pars, **args) -> None:
        """
        Initialize the instance with a given name.
        
        Parameters
        ----------
        name : str, optional
            The name to assign to the instance. Default is "Untitled".
        
        Returns
        -------
        None

        """
        
        try:
            self.name = args["name"]
        except:
            self.name = "Untitled"
        
        self.pars = pars
        # initialise priors as uniform distributions
        [setattr(self, "prior_" + p, distribution(uniform, unif_value=1)) for p in self.pars]
        
        try:
            self.theta_hat = args["theta_hat"]
            self.ihess = args["ihess"]
            self.laplace_posterior()
        except:
            self.theta_hat = None
            self.ihess = None
            print("must run MAP")
        
    def load_prior(self, par: str, dist, **args: Dict) -> None:
        """
        Load a prior distribution for a specified parameter.
    
        Parameters
        ----------
        par : str
            The name of the parameter.
        dist :
            The distribution function or class.
        args : Dict[str]
            Additional arguments to be passed to the distribution function or class.
    
        Returns
        -------
        None

        """
        setattr(self, "prior_" + par, distribution(dist, **args))
        
    def load_log_likelihood(self, log_loss_fn: callable, **args):
        """
        Load a likelihood loss function.

        Parameters
        ----------
        log_loss_fn : callable
            DESCRIPTION.
        **args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.log_likelihood_args = args
        self.log_likelihood_loss = log_loss_fn
    
    @abstractmethod
    def predictor(self, D, *P: Dict) -> None:
        """
        Abstract method for making predictions using a model.
         
        Parameters
        ----------
        D : np.ndarray
            Training Input dataset.
        P : Any
            Trainable parameters.
         
        Returns
        -------
        Any
            The result of the prediction.
    
        """
        ...
    
    def log_prior(self, *P: Dict) -> float:
        """
        Calculate the log-prior probability.
    
        Parameters
        ----------
        P : Dict
            Trainable parameters.
    
        Returns
        -------
        float
            The log-prior probability.

        """
        return np.array([getattr(self, "prior_" + p).logpdf(P[(self.pars.index(p))]) for p in self.pars]).sum()
    
    def log_likelihood(self, D, *P: Dict) -> float:
        """
        Calculate the log-likelihood.
    
        Parameters
        ----------
        D : 
            Input dataset.
        P : Dict[str]
            Trainable parameters.
    
        Returns
        -------
        float
            The log-posterior probability.

        """
        return -self.log_likelihood_loss(D.y, self.predictor(D, *P), **self.log_likelihood_args)
    
    def log_posterior(self, D, *P: Dict) -> float:
        """
        Calculate the log-posterior.
    
        Parameters
        ----------
        D : 
            Input dataset.
        P : Dict[str]
            Trainable parameters.
    
        Returns
        -------
        float
            The log-posterior probability.

        """
        return self.log_prior(*P) + self.log_likelihood(D, *P)
    
    def MAP(self, D, x0=[1,1], solver: str ="L-BFGS-B"):
        
        def callback(X):
            current_min = -self.log_posterior(D, *X)
            print(f"Iter: {self.n_eval:d} -- Params: {X} -- Min {current_min:.3f}")
            self.n_eval += 1
        
        if self.theta_hat is not None and self.ihess is not None:
            print("skipping map")
        else:
            self.n_eval = 0
            result = minimize(lambda t: -self.log_posterior(D, *t), x0=x0, method=solver, callback=callback,
                              options={'disp': True,
                                       'maxiter': 1e10,
                                       'maxls': 1e10,
                                       'gtol': 1e-15, #-15
                                       'ftol': 1e-15, #-15
                                       'eps': 1e-6}, ) #-6
    
            if result.success:
                self.theta_hat = result.x
                self.ihess = result.hess_inv.todense()
                self.laplace_posterior()
            else:
                raise Exception("MAP did not succeede.")
            
    def laplace_posterior(self):
        self.joint = multivariate_normal(mean = self.theta_hat, cov=self.ihess)
        for idx in range(self.theta_hat.shape[0]):
            setattr(self, "marginal_" + self.pars[idx], norm(loc=self.theta_hat[idx], scale=self.ihess[idx][idx]**0.5))
      
    def predictive_posterior(self, posterior_samples, D):
        self.posterior_samples = posterior_samples
        predictions = []
        
        for k in range(0,self.posterior_samples):
            predictions.append(self.predictor(D, *self.joint.rvs(1)))
        
        predictions = np.array(predictions)
        
        return predictions
    
    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"

class AbstractMAPViewer(ABC):
    
    def __init__(self, p1, b1, n1, p2, b2, n2, spacing):
        
        self.pars = (p1, p2)
        self.p1 = p1
        self.p2 = p2
        self.n1 = n1
        self.n2 = n2
        self.spacing = spacing
        setattr(self, "bounds_" + p1, b1)
        setattr(self, "bounds_" + p2, b2)
        
        X1, X2 = grid_factory(getattr(self, "bounds_" + p1),
                              getattr(self, "bounds_" + p2),
                              self.n1, self.n2, spacing)
        setattr(self, p1, X1)
        setattr(self, p2, X2)
    
    @abstractmethod
    def contour(self):
        ...
    
    def config_contour(self):
        pass
    
    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"


class AbstractDataset(ABC):

    def __init__(self, **kwargs):
        self.X = None
        self.y = None

        try:
            path = kwargs.pop("path")
            reader = kwargs.pop("reader")
            self.data = reader(path, **kwargs)

        except KeyError:
            [setattr(self, k, kwargs[k]) for k in kwargs.keys()]

            assert self.X is not None
            assert self.y is not None

        except KeyError:
            self.X = kwargs.pop("X")
            self.y = kwargs.pop("y")

        except KeyError:
            raise Exception("data load unsuccessful.")

    @abstractmethod
    def pre_process():
        pass

    @abstractmethod
    def populate():
        pass

    @abstractmethod
    def partition():
        pass