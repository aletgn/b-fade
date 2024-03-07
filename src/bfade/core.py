from bfade.statistics import distribution
from bfade.elhaddad import ElHaddadCurve
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

from abc import ABC, abstractmethod


class AbstractBayes(ABC):
    
    def __init__(self, name: str = "Untitled"):
        self.name = name
        
    def declare_parameters(self, *pars):
        self.pars = pars
        
    def load_prior(self, par: str, dist, **args):
        setattr(self, "prior_" + par, distribution(dist, **args))
        
    def load_log_likelihood(self, log_loss_fn: callable, **args):
        [setattr(self, a, args[a]) for a in args]
        self.log_likelihood_loss = log_loss_fn
    
    @abstractmethod
    def predictor(self, D, *P):
        ...
    
    def log_prior(self, *P):
        return np.array([getattr(self, "prior_" + p).logpdf(P[(self.pars.index(p))]) for p in self.pars]).sum()
    
    def log_likelihood(self, D, *P):
        return -self.log_likelihood_loss(D.y, self.predictor(D, *P))
    
    def log_posterior(self, D, *P):
        return self.log_prior(*P) + self.log_likelihood(D, *P)
    
    def MAP(self, D, x0=[1,1]):
        
        def callback(X):
            current_min = -self.log_posterior(D, *X)
            print(f"Iter: {self.n_eval:d} -- Params: {X} -- Min {current_min:.3f}")
            self.n_eval += 1
        
        self.n_eval = 0
        result = minimize(lambda t: -self.log_posterior(D, *t), x0=x0, method="L-BFGS-B", callback=callback,
                          options={'disp': True,
                                   'maxiter': 1e10,
                                   'maxls': 1e10,
                                   'gtol': 1e-15,
                                   'ftol': 1e-15,
                                   'eps': 1e-7}, )

        print(result)
        print(result.hess_inv.todense())
    
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
        
        

# class Bayes:
    
#     def __init__(self, name: str = "Untitled"):
#         self.name = name
        
#         # likelihood parameters
#         self.eps = None
#         self.normalize = None
        
#     def declare_parameters(self, *pars):
#         self.pars = pars
        
#     def load_prior(self, par: str, dist, **args):
#         setattr(self, "prior_" + par, distribution(dist, **args))
        
#     def load_log_likelihood(self, log_loss_fn: callable, eps="auto", normalize=False):
#         self.eps = eps
#         self.normalize = normalize
#         self.log_likelihood_loss = log_loss_fn
    
#     def predictor(self, D, *P):
#         eh = ElHaddadCurve(metrics=np.log10, dk_th=P[0], ds_w=P[1], y=0.65)
#         signed_distance, _, _ = eh.signed_distance_to_dataset(D.X)
#         return expit(signed_distance)
    
#     def log_prior(self, *P):
#         return np.array([getattr(self, "prior_" + p).logpdf(P[(self.pars.index(p))]) for p in self.pars]).sum()
    
#     def log_likelihood(self, D, *P):
#         return -self.log_likelihood_loss(D.y, self.predictor(D, *P), eps=self.eps, normalize=self.normalize)
    
#     def log_posterior(self, D, *P):
#         return self.log_prior(*P) + self.log_likelihood(D, *P)
    
#     def __repr__(self):
#         attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
#         return f"{self.__class__.__name__}({attributes_str})"