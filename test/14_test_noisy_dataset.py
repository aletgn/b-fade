#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath

syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from sklearn.metrics import log_loss

from bfade.dataset import Dataset, SyntheticDataset
from bfade.elhaddad import ElHaddadDataset
from bfade.elhaddad import ElHaddadCurve
from bfade.elhaddad import ElHaddadBayes
from bfade.viewers import PreProViewer, BayesViewer

eh = ElHaddadCurve(dk_th=8, ds_w=1100, y=0.8)

sd = SyntheticDataset()
sd.make_grid([1,1000], [200, 1500], 20, 20, spacing="log")
sd.make_classes(eh)
# sd.inspect([0.5, 1500], [100, 2000], scale="log", curve=eh, x=np.linspace(1,1000,1000))
sd.add_noise(50,500)
sd.crop_points()
sd.inspect([0.5, 1500], [100, 2000], scale="log", curve=eh, x=np.linspace(1,1000,1000))

bay = ElHaddadBayes("dk_th", "ds_w", y=0.8)
bay.load_log_likelihood(log_loss, normalize=False)

v = BayesViewer("dk_th", [9,13], 15, "ds_w", [1100, 1500], 15)
v.contour("log_likelihood", bay, sd)

bay.MAP(sd, x0=[5, 500])

