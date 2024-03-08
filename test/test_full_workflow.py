from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from bfade.elhaddad import ElHaddadCurve
from bfade.datagen import SyntheticDataset
from bfade.core import BayesElHaddad

from sklearn.metrics import log_loss
from scipy.stats import norm

def invoke_curve():
    eh = ElHaddadCurve(dk_th=3, ds_w=400, y=0.73, name = "EH test", metrics=np.log10)
    # eh.inspect(np.linspace(1,1000, 1000), scale="log")
    return eh

def gen_data(curve):
    sd = SyntheticDataset(eh)
    sd.make_grid([1, 1000],[50, 800], 20, 20, spacing="log")
    sd.clear_points(tol=10)
    sd.make_classes()
    # sd.inspect(np.linspace(1,1000,1000), scale="log")
    return sd

def signed_distance(curve, dataset):
    signed_dist, x1_min, x2_min = curve.signed_distance_to_dataset(dataset.X)
    curve.inspect_signed_distance(np.linspace(1, 1000, 100), x1_min, x2_min, signed_dist, dataset.X, scale="log")

def bayesian_inference():
    b = BayesElHaddad("dk_th", "ds_w")
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    b.load_log_likelihood(log_loss, normalize=True)

if __name__ == "__main__":
    eh = invoke_curve()
    sd = gen_data(eh)
    #signed_distance(eh, sd)
    
    
    
    