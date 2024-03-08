import numpy as np
from typing import Dict, List

from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import t as t_student

from bfade.statistics import distribution

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