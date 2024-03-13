from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bfade.datagen import SyntheticDataset
from bfade.elhaddad import ElHaddadCurve
from bfade.elhaddad import ElHaddadBayes
from bfade.statistics import uniform
from bfade.util import grid_factory
from bfade.viewers import BayesViewer, LaplacePosteriorViewer

from sklearn.metrics import log_loss


def istantiation():
    v = BayesViewer("dk_th", [3, 7], 5,
                "ds_w", [400, 800], 5,
                spacing="lin")
    
    print(v)

def view_bayes():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, y=0.65)
    
    d = SyntheticDataset(eh)
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(tol=1)
    d.make_classes()
    # d.inspect(np.linspace(1, 1000, 1000), scale="log")
    
    b = ElHaddadBayes("dk_th", "ds_w", y=0.65)
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    b.load_log_likelihood(log_loss, normalize=True)
    
    v = BayesViewer("dk_th", [3, 7], 5,
                    "ds_w", [400, 800], 5,
                    spacing="lin")
    
    v.contour("log_prior", b)
    v.contour("log_likelihood", b, d)
    v.contour("log_posterior", b, d)

def view_laplace():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, y=0.65)
    
    d = SyntheticDataset(eh)
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(tol=1)
    d.make_classes()
    # d.inspect(np.linspace(1, 1000, 1000), scale="log")
    
    b = ElHaddadBayes("dk_th", "ds_w", theta_hat=np.array([5.00663972,600.18485208]),
                      ihess=np.array([[ 5.06170001e-01, -1.01078680e+01],   [-1.01078680e+01,  1.40680324e+03]]))
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    b.load_log_likelihood(log_loss, normalize=True)
    
    
    l = LaplacePosteriorViewer("dk_th", 2, 10, "ds_w", 2, 10, bayes=b)
    l.contour(b)
    # l.marginals("dk_th", b)
    # l.marginals("ds_w", b)
    print(l)
    # 
    # print(b)
    print(eh)

if __name__ == "__main__":
    istantiation()
    view_bayes()
    view_laplace()
    pass
