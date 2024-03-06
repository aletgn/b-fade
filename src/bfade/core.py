from bfade.statistics import distribution
from bfade.elhaddad import ElHaddadCurve
import numpy as np

def logistic_predictor(P):
    pass



class Bayes:
    
    def __init__(self, name: str = "Untitled"):
        self.name = name
        
        # likelihood parameters
        self.eps = None
        self.normalise = None
        
    def declare_parameters(self, *pars):
        self.pars = pars
        
    def load_prior(self, par: str, dist, **args):
        setattr(self, "prior_" + par, distribution(dist, **args))
        
    def load_log_likelihood(self, log_loss: callable, eps="auto", normalize=False):
        self.eps = eps
        self.normalise = normalize
        self.log_likelihood_loss = log_loss
    
    def predictor(self, P):
        pass        
    
    def log_prior(self, P):
        return np.array([getattr(self, "prior_" + p).logpdf(P[(self.pars.index(p))]) for p in self.pars]).sum()
    
    def log_likelihood(self, P, X, y):
        pass
    
    def log_posterior(self, P, X, y):
        return self.log_prior(P) + self.log_likelihood(P, X)
    
    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"