from bfade.statistics import distribution
from bfade.elhaddad import ElHaddadCurve
import numpy as np
from typing import Dict

from scipy.special import expit
from scipy.optimize import minimize

from abc import ABC, abstractmethod


class AbstractBayes(ABC):
    """Bayesian framework to perform Maximum a Posterior Estimation."""
    
    def __init__(self, name: str = "Untitled") -> None:
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
        self.name = name
        
    def declare_parameters(self, *pars: Dict) -> None:
        """
        Declare and assign parameters to the instance.
    
        Parameters
        ----------
        pars : Dict[str]
            Variable-length argument list of dictionaries representing parameters.
    
        Returns
        -------
        None

        """
        self.pars = pars
        
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
        [setattr(self, a, args[a]) for a in args]
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
        return -self.log_likelihood_loss(D.y, self.predictor(D, *P))
    
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
        
        self.n_eval = 0
        result = minimize(lambda t: -self.log_posterior(D, *t), x0=x0, method=solver, callback=callback,
                          options={'disp': True,
                                   'maxiter': 1e10,
                                   'maxls': 1e10,
                                   'gtol': 1e-15,
                                   'ftol': 1e-15,
                                   'eps': 1e-6}, )

        if result.success:
            self.x_hat = result.x
            self.ihess = result.hess_inv.todense()
        else:
            raise Exception("MAP did not succeede.")
    
    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"


class BayesElHaddad(AbstractBayes):
    
    def __init__(self):    
        super().__init__()
        
    def predictor(self, D, *P):
        eh = ElHaddadCurve(metrics=np.log10, dk_th=P[0], ds_w=P[1], y=0.65)
        signed_distance, _, _ = eh.signed_distance_to_dataset(D.X)
        return expit(signed_distance)
