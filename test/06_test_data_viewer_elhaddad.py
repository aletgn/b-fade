from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bfade.dataset import SyntheticDataset
from bfade.elhaddad import ElHaddadBayes, ElHaddadCurve, ElHaddadDataset
from bfade.statistics import uniform, MonteCarlo
from bfade.util import grid_factory, logger_manager
from bfade.viewers import BayesViewer, LaplacePosteriorViewer, PreProViewer
import pandas as pd

logger_manager(level="DEBUG")

def curves():
    pp = PreProViewer(scale="log")
    
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    eh1 = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=500, Y=0.65)
    pp.view(curve=[eh, eh1])
    
def pi():
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65, theta_hat=np.array([5.00663972,600.18485208]),
                      ihess=np.array([[ 5.06170001e-01, -1.01078680e+01],   [-1.01078680e+01,  1.40680324e+03]]))

    m = MonteCarlo(ElHaddadCurve)
    m.sample(1000, "joint", b)
    m.sample(1000, "marginals", b)
    # m.prediction_interval([1,1000], 1000, "lin", ElHaddadCurve, Y=0.65)
    
    pp = PreProViewer([1,1000], [100,700], 1000, "log")
    pp.view(prediction_interval=m,
            mc_samples = 100,
            mc_bayes=b,
            mc_distribution="joint",
            confidence=95)
    
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    eh1 = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=500, Y=0.65)
    pp.view(curve = [eh, eh1], prediction_interval=m,
            mc_samples = 100,
            mc_bayes=b,
            mc_distribution="joint",
            confidence=95)
    
def pred_post():
    eh = ElHaddadCurve(dk_th=5, ds_w=600, Y=0.65)
    eh1 = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=500, Y=0.65)
    # eh.inspect(np.linspace(1, 1000, 1000), scale="log")
    
    d = SyntheticDataset()
    d.make_grid([1, 1000],[100, 700], 15, 15, spacing="log")
    # d.inspect(scale="log")
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65,
                      theta_hat=np.array([5.00663972,600.18485208]),
                      ihess=np.array([[ 5.06170001e-01, -1.01078680e+01],
                                      [-1.01078680e+01,  1.40680324e+03]]))
    
    m = MonteCarlo(ElHaddadCurve)
    m.sample(1000, "joint", b)
    m.sample(1000, "marginals", b)
    
    pp = PreProViewer([1,1000], [100,700], 1000, "log", Y=0.65)
    pp.view(predictive_posterior=b, post_samples = 10, post_data=d, post_op=np.mean, curve = [eh, eh1])

def data_view():
    a = ElHaddadDataset(reader=pd.read_csv, path="/home/ale/Downloads/SyntheticEH.csv")
    data = a.pre_process()
    train, test = a.partition("user")
    pp = PreProViewer([1,1000], [100,700], 1000, "log")
    pp.config(save=False, folder="/home/ale/Desktop/plots/", fmt="png", dpi=300)
    pp.config_canvas(class0="Runout", class1="Failed", xlabel="sqrt_area", ylabel="delta_sigma", cbarlabel="delta_k")
    pp.view(train_data=train, test_data=test)


if __name__ == "__main__":
    # curves()
    # pi()
    # pred_post()
    # data_view()
    pass