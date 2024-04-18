#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np

import sklearn.metrics

from scipy.stats import norm

from bfade.elhaddad import ElHaddadCurve, ElHaddadBayes
from bfade.elhaddad import ElHaddadTranslator as ET
from bfade.dataset import SyntheticDataset
from bfade.viewers import BayesViewer, LaplacePosteriorViewer, PreProViewer
from bfade.util import parse_arguments, get_config_file, config_matplotlib
from bfade.statistics import MonteCarlo
config_matplotlib(font_size=10, font_family="sans-serif", use_latex=False, interactive=False)

# %% Istantiate El Haddad curve
eh = ElHaddadCurve(dk_th=4, ds_w=100, Y=0.46, metrics=np.log10, name="EH_4_100")
# eh.inspect(np.linspace(1,1000, 1000), scale="log")

# %% Generate dataset
sd = SyntheticDataset(name="EH_4_100")
sd.make_tube(eh, x_bounds=[1,2000], n=30, up=0.02, down=-0.02, step=8, spacing="log")
# sd.clear_points(tol=2)
sd.make_classes(eh)
# sd.config(save=cf["save"], folder=cf["pic_folder"])
# sd.inspect(np.linspace(1, 2000, 1000), scale="log")
# signed_dist, x1_min, x2_min = eh.signed_distance_to_dataset(sd)
# eh.inspect_signed_distance(np.linspace(1, 1000, 100), x1_min, x2_min, signed_dist, sd.X, scale="log")

# %% Baysian Inference
bay = ElHaddadBayes("dk_th", "ds_w", Y=0.46, name="EH_4_100")
bay.load_log_likelihood(sklearn.metrics.log_loss, normalize=False)
# bay.load_prior("dk_th", norm, loc=1, scale=0.1)
# bay.load_prior("ds_w", norm, loc=250, scale=10)

v = BayesViewer("dk_th", [2,6], 10, "ds_w", [70,120], 10, name="EH_4_100")
v.config_contour(translator=ET)
# v.contour("log_prior", bay)
# v.contour("log_likelihood", bay, sd)
# v.contour("log_posterior", bay, sd
bay.MAP(sd, x0=[3, 120])

theta_hat = np.array([3.9972668, 100.01953756])
ihess = np.array([[ 0.00116242, -0.00203901], [-0.00203901, 0.05873539]])

bay_id = ElHaddadBayes("dk_th", "ds_w", theta_hat=theta_hat, ihess=ihess, Y=0.46, name="EH_2_250")

ll = LaplacePosteriorViewer("dk_th", 4, 10, "ds_w", 4, 10, bayes=bay_id)
ll.config_contour(translator=ET)
# ll.contour(bay_id)
# ll.marginals("dk_th", bay_id)
# ll.marginals("ds_w", bay_id)

mc = MonteCarlo(ElHaddadCurve)
mc.sample(1000, "marginals", bay_id)

# mc.sample_joint(bay_id)
# mc.sample_marginals(bay_id)
mean, pred, _ = mc.prediction_interval([1,1000], 1000, "lin", 95)


pp = PreProViewer([1,1500], [30, 1200], 1000, scale="log")
pp.config_canvas(xlabel="sq_a", ylabel="ds", cbarlabel="dk", translator=ET)
# pp.view(train_data=sd)
# pp.view(test_data=sd)
# pp.view(curve=[eh])
pp.view(prediction_interval=mc,
        mc_samples = 100,
        mc_bayes=bay_id,
        mc_distribution="joint", confidence=99)
# pp.view(predictive_posterior=bay_id, post_samples=5, post_data=sd, 
#         post_op = np.mean, curve=[eh], prediction_interval=mc, train_data=sd)