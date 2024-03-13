from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bfade.elhaddad import ElHaddadBayes, ElHaddadCurve
from bfade.statistics import uniform, MonteCarlo
from bfade.viewers import BayesViewer, LaplacePosteriorViewer

def monte_carlo():

    b = ElHaddadBayes("dk_th", "ds_w", y=0.65, theta_hat=np.array([5.00663972,600.18485208]),
                    ihess=np.array([[ 5.06170001e-01, -1.01078680e+01],   [-1.01078680e+01,  1.40680324e+03]]))

    m = MonteCarlo(ElHaddadCurve)
    m.sample(1000, "joint", b)
    m.sample(1000, "marginals", b)
    m.prediction_interval([1,1000], 1000, "lin", 95)

if __name__ == "__main__":
    monte_carlo()
