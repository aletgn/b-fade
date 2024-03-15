#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath

syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import norm

from bfade.statistics import MonteCarlo
from bfade.dataset import Dataset, SyntheticDataset
from bfade.elhaddad import ElHaddadDataset
from bfade.elhaddad import ElHaddadCurve
from bfade.elhaddad import ElHaddadBayes
from bfade.viewers import PreProViewer, BayesViewer, LaplacePosteriorViewer
from bfade.util import save


eh = ElHaddadCurve(dk_th=8, ds_w=1100, Y=0.8)

sd = SyntheticDataset()
sd.make_grid([1,1000], [200, 1500], 20, 20, spacing="log")
sd.make_classes(eh)
# sd.inspect([0.5, 1500], [100, 2000], scale="log", curve=eh, x=np.linspace(1,1000,1000))
sd.add_noise(50,500)
sd.crop_points()
# sd.inspect([0.5, 1500], [100, 2000], scale="log", curve=eh, x=np.linspace(1,1000,1000))

bay = ElHaddadBayes("dk_th", "ds_w", Y=0.8, name="BayesNotReg")
bay.load_log_likelihood(log_loss, normalize=False)

v = BayesViewer("dk_th", [7,13], 15, "ds_w", [900, 1500], 15)
# v.contour("log_likelihood", bay, sd)

bay_reg = ElHaddadBayes("dk_th", "ds_w", Y=0.8, name="BayesReg")
bay_reg.load_log_likelihood(log_loss, normalize=True)
bay_reg.load_prior("dk_th", norm, loc=8, scale=1)
bay_reg.load_prior("ds_w", norm, loc=1100, scale=50)
# v.contour("log_prior", bay_reg)
# v.contour("log_posterior", bay_reg, sd)

bay.MAP(sd, x0=[5, 500])
bay_reg.MAP(sd, x0=[5, 500])

# bay = ElHaddadBayes("dk_th", "ds_w", Y=0.8,
#                     theta_hat=np.array([11.00317884, 1357.16796931]),
#                     ihess=np.array([[3.50694209e-03, -3.11428133e-01], [-3.11428133e-01,  5.13663539e+01]]),
#                     name="BayesNotReg")

# bay_reg = ElHaddadBayes("dk_th", "ds_w", Y=0.8,
#                         theta_hat= np.array([8.49759908, 1105.07891879]),
#                         ihess= np.array([[9.08663233e-01, 4.65971142e+00], [4.65971142e+00, 2.34423343e+03]]),
#                         name="BayesReg")

# l = LaplacePosteriorViewer("dk_th", 4, 50, "ds_w", 4, 50, bay)
# l_reg = LaplacePosteriorViewer("dk_th", 4, 50, "ds_w", 4, 50, bay_reg)
# l.contour(bay)
# l.contour(bay_reg)

opt = ElHaddadCurve(dk_th=bay.theta_hat[0], ds_w=bay.theta_hat[1], Y=0.8, name="NotReg")
opt_reg = ElHaddadCurve(dk_th=bay_reg.theta_hat[0], ds_w=bay_reg.theta_hat[1], Y=0.8, name="Reg")
mc = MonteCarlo(ElHaddadCurve)

evald = SyntheticDataset()
evald.make_grid([1,1000], [200, 1500], 20, 20, spacing="log")

p = PreProViewer([1,1000], [200, 1500], 1000, scale="log")

p.view(train_data=sd, curve=[eh, opt_reg, opt], prediction_interval=mc,
       mc_samples=1000, mc_bayes=bay_reg, mc_distribution="joint", confidence=95,
       predictive_posterior=bay_reg, post_samples=10, post_data=evald, post_op=np.mean)

point = Dataset(X=np.array([[500, 200]]))
pred = bay_reg.predictive_posterior(10, point, np.mean)
print(pred)

pred = bay.predictor(point, bay.theta_hat[0], bay.theta_hat[1])
print(pred)

save(bay, bay_reg, folder="./")