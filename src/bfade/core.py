from bfade.statistics import distribution
from bfade.elhaddad import ElHaddadCurve
import numpy as np
from typing import Dict, List

from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import t as t_student

from abc import ABC, abstractmethod


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
                                       'gtol': 1e-15,
                                       'ftol': 1e-15,
                                       'eps': 1e-6}, )
    
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


class BayesElHaddad(AbstractBayes):
    
    def __init__(self, *pars, **args):    
        super().__init__(*pars, **args)
        
    def predictor(self, D, *P):
        eh = ElHaddadCurve(metrics=np.log10, dk_th=P[0], ds_w=P[1], y=0.65)
        signed_distance, _, _ = eh.signed_distance_to_dataset(D.X)
        return expit(signed_distance)


class MonteCarlo:
    
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample_joint(self, bayes):
        self.pars = bayes.pars
        self.samples = bayes.joint.rvs(self.n_samples)
    
    def sample_marginals(self, bayes):
        self.pars = bayes.pars
        self.samples = np.array([getattr(bayes, "marginal_" + p).rvs(self.n_samples) for p in bayes.pars]).T
        
    def prediction_interval(self, x_edges, n, spacing, curve, confidence=0.95, **args):
        curves = []
        if spacing == "log":
            x1 = np.logspace(np.log10(x_edges[0]), np.log10(x_edges[1]), n)
        else:
            x1 = np.linspace(x_edges[0], x_edges[1], n)
        
        # fig, ax = plt.subplots(dpi=300)
        
        for s in self.samples:
            d = dict(zip(self.pars, s))
            d.update(**args)
            curves.append(curve(**d).equation(x1))
            # ax.loglog(x1, curve(**d).equation(x1))
            
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        pred = t_student(df=len(curves)-1).ppf(confidence/100)*std*((1+1/len(curves))**0.5)
        
        return mean, pred