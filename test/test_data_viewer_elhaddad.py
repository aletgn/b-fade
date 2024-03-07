from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bfade.datagen import SyntheticDataset
from bfade.elhaddad import ElHaddadCurve
from bfade.core import BayesElHaddad, MonteCarlo
from bfade.statistics import uniform
from bfade.util import grid_factory
from bfade.viewers import BayesViewer, LaplacePosteriorViewer, PreProViewer

def curves():
    pp = PreProViewer(scale="log")
    
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, y=0.65)
    eh1 = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=500, y=0.65)
    pp.view(curve=[eh, eh1])
    
def pi():
    b = BayesElHaddad("dk_th", "ds_w", theta_hat=np.array([5.00663972,600.18485208]),
                      ihess=np.array([[ 5.06170001e-01, -1.01078680e+01],   [-1.01078680e+01,  1.40680324e+03]]))

    m = MonteCarlo(1000)
    m.sample_joint(b)
    # m.sample_marginals(b)
    # m.prediction_interval([1,1000], 1000, "lin", ElHaddadCurve, y=0.65)
    
    pp = PreProViewer([1,1000], [100,700], 1000, "log", "y")
    pp.view(prediction_interval=m, ref_curve=ElHaddadCurve, confidence=0.95, y=0.65)
    
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, y=0.65)
    eh1 = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=500, y=0.65)
    pp.view(curve = [eh, eh1], prediction_interval=m, ref_curve=ElHaddadCurve, confidence=0.95, y=0.65)
    
def pred_post():
    pass



if __name__ == "__main__":
    # curves()
    # pi()
    pred_post()