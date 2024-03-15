from sys import path as syspath
from os import path as ospath

syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np

from bfade.util import load
from bfade.dataset import SyntheticDataset, Dataset
from bfade.viewers import PreProViewer
from bfade.elhaddad import ElHaddadBayes

sd = SyntheticDataset()
sd.make_grid([1,1000], [200, 1500], 20, 20, spacing="log")

bay, bay_reg = load(folder="./", extension="bfd")
p = PreProViewer([1,1000], [200, 1500], 1000, scale="log")
p.view(predictive_posterior=bay_reg, post_samples=10, post_data=sd, post_op=np.mean)

point = Dataset(X=np.array([[500, 200]]))
pred = bay_reg.predictive_posterior(10, point, np.mean)
print(pred)

pred = bay.predictor(point, bay.theta_hat[0], bay.theta_hat[1])
print(pred)


pred1 = bay.predict(point)
print(pred1)

# must not work -- did so to test exception in predict method (wrapper)
bay_exc = ElHaddadBayes("dk_th", "ds_w", Y=0.8)
bay_exc.predict(point)