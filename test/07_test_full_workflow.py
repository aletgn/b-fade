from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
# np.seterr(divide='ignore', invalid='ignore')

from bfade.elhaddad import ElHaddadCurve, ElHaddadBayes
from bfade.dataset import SyntheticDataset
from bfade.viewers import BayesViewer, LaplacePosteriorViewer
from bfade.statistics import MonteCarlo

# change the bound of the distance minimizer
# import bfade
# bfade.abstract.MINIMZER_LO_BOUND = aFloat
# bfade.abstract.MINIMZER_UP_BOUND = aFloat

from sklearn.metrics import log_loss
from scipy.stats import norm


def invoke_curve():
    eh = ElHaddadCurve(dk_th=3, ds_w=400, Y=0.73, name = "EH test", metrics=np.log10)
    # eh.inspect(np.linspace(1,1000, 1000), scale="log")
    return eh

def gen_data(curve):
    sd = SyntheticDataset()
    sd.make_grid([1, 1000],[50, 800], 35, 35, spacing="log")
    sd.clear_points(curve, tol=1)
    sd.make_classes(curve)
    # sd.inspect(np.linspace(1,1000,1000), scale="log")
    return sd

def signed_distance(curve, dataset):
    signed_dist, x1_min, x2_min = curve.signed_distance_to_dataset(dataset)
    curve.inspect_signed_distance(np.linspace(1, 1000, 100), x1_min, x2_min, signed_dist, dataset.X, scale="log")

def bayesian_inference(dataset):
    bay = ElHaddadBayes("dk_th", "ds_w", Y=0.73)
    # bay.load_prior("dk_th", norm, loc=5, scale=1)
    # bay.load_prior("ds_w", norm, loc=600, scale=50)
    bay.load_log_likelihood(log_loss, normalize=False)

    # v = BayesViewer("dk_th", [1, 5], 2, "ds_w", [200, 600], 2, spacing="lin")
    # v.contour("log_prior", bay)
    # v.contour("log_likelihood", bay, dataset)
    # v.contour("log_posterior", bay, dataset)
    
    bay.MAP(dataset, x0=[2, 200])
    print(bay.theta_hat, bay.ihess)
    return bay

def laplace_view():
    bay = ElHaddadBayes("dk_th", "ds_w", Y=0.73, theta_hat = np.array([3.00243386, 400.00344183]),
                        ihess = np.array([[ 1.84780103e-03, -5.94696958e-02],
                                          [-5.94696958e-02,  1.32796603e+01]]))
    
    l = LaplacePosteriorViewer("dk_th", 4, 10, "ds_w", 4, 10, bayes=bay)
    l.contour(bay)
    l.marginals("ds_w", bay)

def monte_carlo():
    bay = ElHaddadBayes("dk_th", "ds_w", Y=0.65,
                        theta_hat = np.array([3.00243386, 400.00344183]),
                        ihess = np.array([[ 1.84780103e-03, -5.94696958e-02],
                                          [-5.94696958e-02,  1.32796603e+01]]))

    mc = MonteCarlo(ElHaddadCurve)
    mc.sample(1000, "marginals", bay)
    mean, pred, _ = mc.prediction_interval([1,1000], 1000, "lin", 95)
    print(mean, pred)

if __name__ == "__main__":
    # eh = invoke_curve()
    # sd = gen_data(eh)
    # signed_distance(eh, sd)
    # bay = bayesian_inference(sd)
    # laplace_view()
    # monte_carlo()
    pass
    
    
    
    