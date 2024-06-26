from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np

from bfade.elhaddad import ElHaddadCurve, ElHaddadBayes
from bfade.dataset import SyntheticDataset
from bfade.viewers import BayesViewer, LaplacePosteriorViewer
from bfade.statistics import MonteCarlo

from sklearn.metrics import log_loss
from scipy.stats import norm

from bfade.util import config_matplotlib

config_matplotlib(font_size=12, font_family="sans-serif", use_latex="False")

def invoke_curve():
    eh = ElHaddadCurve(dk_th=3, ds_w=400, Y=0.73, name = "EH test", metrics=np.log10)
    # eh.inspect(np.linspace(1,1000, 1000), scale="log")
    return eh

def gen_data(curve):
    sd = SyntheticDataset()
    sd.make_grid([1, 1000],[50, 800], 30, 30, spacing="log")
    sd.clear_points(curve, tol=1)
    sd.make_classes(curve)
    # sd.inspect(np.linspace(1,1000,1000), scale="log")
    return sd

def bayesian_view(dataset):
    bay = ElHaddadBayes("dk_th", "ds_w", Y=0.73)
    
    bay.load_log_likelihood(log_loss, normalize=True)
    bay.load_prior("dk_th", norm, loc=5, scale=1)
    bay.load_prior("ds_w", norm, loc=600, scale=50)
    bay.MAP(dataset, x0=[2, 300])
    v = BayesViewer("dk_th", [1, 5], 2,
                    "ds_w", [200, 600], 2, spacing="lin",
                    name="testBay")

    v.config_contour(cmap="RdYlBu_r")
    # v.contour("log_prior", bay)
    # v.contour("log_likelihood", bay, dataset)
    # v.contour("log_posterior", bay, dataset)

def laplace_view():
    bay = ElHaddadBayes("dk_th", "ds_w", Y=0.73,
                        theta_hat = np.array([3.20417631, 575.33348794]),
                        ihess = np.array([[ 2.26201184e-01, -4.54082872e+00],
                                          [-4.54082872e+00,  1.71398957e+03]]))

    l = LaplacePosteriorViewer("dk_th", 4, 50, "ds_w", 4, 50, bayes=bay)
    l.contour(bay)
    l.marginals("ds_w", bay)
    l.marginals("dk_th", bay)

if __name__ == "__main__":
    # eh = invoke_curve()
    # sd = gen_data(eh)
    # bayesian_view(sd)
    # laplace_view()
    pass