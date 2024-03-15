from sys import path as syspath
from os import path as ospath

syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np

from bfade.util import load
from bfade.dataset import SyntheticDataset
from bfade.viewers import PreProViewer

sd = SyntheticDataset()
sd.make_grid([1,1000], [200, 1500], 20, 20, spacing="log")

bay, bay_reg = load(folder="./", extension="bfd")

p = PreProViewer([1,1000], [200, 1500], 1000, scale="log")
p.view(predictive_posterior=bay_reg, post_samples=10, post_data=sd, post_op=np.mean)