#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import sklearn.metrics

from bfade.elhaddad import ElHaddadCurve, ElHaddadBayes
from bfade.datagen import SyntheticDataset
from bfade.viewers import BayesViewer, LaplacePosteriorViewer, PreProViewer
from bfade.util import parse_arguments, get_config_file, config_matplotlib
from bfade.statistics import MonteCarlo

config_matplotlib(font_size=14, font_family="serif", use_latex=True, interactive=True)
cf = get_config_file(parse_arguments("./09_test_elhaddad.yaml"))

#%% Istantiate El Haddad curve
eh = ElHaddadCurve(dk_th=cf["curve"]["dk_th"], ds_w=cf["curve"]["ds_w"], y=cf["curve"]["y"],
                   metrics=getattr(np, cf["curve"]["metrics"]), name=cf["id"])
eh.config(save=cf["save"], folder=cf["pic_folder"])
eh.inspect(np.linspace(1,1000, 1000), scale="log")

#%% Generate dataset based on the given curve
sd = SyntheticDataset(eh, name=cf["id"])
sd.make_grid(cf["dataset"]["x1"], cf["dataset"]["x2"],
             cf["dataset"]["n1"], cf["dataset"]["n2"],
             spacing=cf["dataset"]["spacing"])
sd.clear_points(tol=cf["dataset"]["tol"])
sd.make_classes()
sd.config(save=cf["save"], folder=cf["pic_folder"])
sd.inspect(np.linspace(1, 1000, 1000), scale="log")

signed_dist, x1_min, x2_min = eh.signed_distance_to_dataset(sd)
eh.inspect_signed_distance(np.linspace(1, 1000, 100), x1_min, x2_min, signed_dist, sd.X, scale="log")

#%% Bayesian Inference -- uniform priors: MAP --> MLE
bay = ElHaddadBayes(cf["bayes"]["p1"], cf["bayes"]["p2"], name=cf["id"])
bay.load_log_likelihood(getattr(sklearn.metrics, cf["bayes"]["log_likelihood"]),
                        normalize=cf["bayes"]["log_normalise"])
v = BayesViewer(cf["bayes"]["p1"], cf["bayes"]["x1"], cf["bayes"]["n1"],
                cf["bayes"]["p2"], cf["bayes"]["x2"], cf["bayes"]["n2"], name=cf["id"])
v.config(save=cf["save"], folder=cf["pic_folder"])
v.contour("log_likelihood", bay, sd)
v.contour("log_posterior", bay, sd)

bay.MAP(sd, x0=cf["map"]["guess"], solver=cf["map"]["solver"])

#%% Display approximated posterior
l = LaplacePosteriorViewer(cf["laplace"]["p1"], cf["laplace"]["c1"], cf["laplace"]["n1"],
                           cf["laplace"]["p2"], cf["laplace"]["c2"], cf["laplace"]["n2"],
                           bayes=bay, name=cf["id"])
l.config(save=cf["save"], folder=cf["pic_folder"])
l.contour(bay)
l.marginals("dk_th", bay)
l.marginals("ds_w", bay)

# %% Monte Carlo
mc = MonteCarlo(cf["montecarlo"]["samples"], ElHaddadCurve, cf["montecarlo"]["confidence"])
mc.sample_joint(bay)
#mean, pred, _ = mc.prediction_interval([1,1000], 1000, "lin", y=0.65)

# %% Pre- and Post-processing 
p = PreProViewer(cf["prepro"]["x_edges"], cf["prepro"]["y_edges"], 
                 cf["prepro"]["n"], cf["prepro"]["scale"],
                 y=cf["prepro"]["y"], name=cf["id"])
p.config(save=cf["save"], folder=cf["pic_folder"])
p.view(train_data=sd)
p.view(test_data=sd)
p.view(train_data=sd, curve=[eh])
p.view(train_data=sd, curve=[eh], prediction_interval=mc)
p.view(predictive_posterior=bay, post_samples = 10, post_data=sd, post_op=np.mean, curve = [eh])


print(bay.theta_hat, bay.ihess)