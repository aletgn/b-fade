#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import sklearn.metrics

from scipy.stats import norm

from bfade.elhaddad import ElHaddadCurve, ElHaddadBayes
from bfade.datagen import SyntheticDataset
from bfade.viewers import BayesViewer, LaplacePosteriorViewer, PreProViewer
from bfade.util import parse_arguments, get_config_file, config_matplotlib
from bfade.statistics import MonteCarlo
config_matplotlib(font_size=14, font_family="sans-serif", use_latex=False, interactive=False)

# %% Istantiate El Haddad curve
eh = ElHaddadCurve(dk_th=2, ds_w=250, y=0.5, metrics=np.log10, name="EH_2_250")
# eh.inspect(np.linspace(1,1000, 1000), scale="log")

# %% Generate dataset
sd = SyntheticDataset(eh, name="EH_2_250")
sd.make_grid([1,1500], [30, 1200], 30, 30, spacing="log")
sd.clear_points(tol=2)
sd.make_classes()
# sd.config(save=cf["save"], folder=cf["pic_folder"])
# sd.inspect(np.linspace(1, 1000, 1000), scale="log")
# signed_dist, x1_min, x2_min = eh.signed_distance_to_dataset(sd)
# eh.inspect_signed_distance(np.linspace(1, 1000, 100), x1_min, x2_min, signed_dist, sd.X, scale="log")

# %% Baysian Inference
# bay = ElHaddadBayes("dk_th", "ds_w", y=0.5, name="EH_2_250")
# bay.load_log_likelihood(sklearn.metrics.log_loss, normalize=False)
#bay.load_prior("dk_th", norm, loc=1, scale=0.1)
#bay.load_prior("ds_w", norm, loc=250, scale=10)

# v = BayesViewer("dk_th", [1,3], 10, "ds_w", [100,300], 10, name="EH_2_250")
# v.contour("log_prior", bay)
# v.contour("log_likelihood", bay, sd)
# v.contour("log_posterior", bay, sd
# bay.MAP(sd, x0=[1, 200])

theta_hat = np.array([2.00438532, 252.69152123])
ihess = np.array([[3.30879748e-03,-4.90809505e-02],
                  [-4.90809505e-02,  8.68712010e+00]])

bay_id = ElHaddadBayes("dk_th", "ds_w", theta_hat=theta_hat, ihess=ihess, y=0.5, name="EH_2_250")

mc = MonteCarlo(100, ElHaddadCurve, 95)
mc.sample_joint(bay_id)
# mc.sample_marginals(bay_id)
# mc.prediction_interval([1,1000], 1000, "lin")

# pred = bay_id.predictive_posterior(5, sd, np.mean, axis=0)


pp = PreProViewer([1,1500], [30, 1200], 1000, scale="log")

# pp.view(train_data=sd)
# pp.view(test_data=sd)
# pp.view(curve=[eh])
# pp.view(prediction_interval=mc)
pp.view(predictive_posterior=bay_id, post_samples=5, post_data=sd,
        post_op = np.mean, curve=[eh])
