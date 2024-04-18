#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

# Erase the line above if you installed the package.

import numpy as np
import sklearn.metrics

from bfade.elhaddad import ElHaddadCurve, ElHaddadBayes
from bfade.dataset import SyntheticDataset
from bfade.viewers import BayesViewer, PreProViewer
from bfade.util import parse_arguments, get_config_file, config_matplotlib, logger_manager

cf = get_config_file(parse_arguments("./eh_shell.yaml"))

logger_manager(level=cf["logger"]["level"])

config_matplotlib(font_size=cf["matplotlib"]["font_size"],
                  font_family=cf["matplotlib"]["font_family"],
                  use_latex=cf["matplotlib"]["use_latex"],
                  interactive=cf["matplotlib"]["interactive"])

# Istantiate the ground truth EH curve
eh = ElHaddadCurve(metrics=getattr(np, cf["curve"]["metrics"]),
                   dk_th=cf["curve"]["dk_th"],
                   ds_w=cf["curve"]["ds_w"],
                   Y=cf["curve"]["Y"],
                   name=cf["name"])
eh.config(save=cf["export"]["save"], folder=cf["export"]["folder"])
eh.inspect(np.linspace(cf["inspect_eh"]["start"],
                       cf["inspect_eh"]["stop"],
                       cf["inspect_eh"]["step"]),
                       scale=cf["inspect_eh"]["scale"])

# Generate training dataset
sd = SyntheticDataset(name=cf["name"])
sd.config(save=cf["export"]["save"], folder=cf["export"]["folder"])
sd.make_grid(cf["train"]["x1"], cf["train"]["x2"],
             cf["train"]["n1"], cf["train"]["n2"],
             spacing=cf["train"]["spacing"])
sd.make_classes(eh)
sd.inspect(cf["train"]["x1"], cf["train"]["x2"], scale=cf["train"]["spacing"],
           curve=eh, x=np.linspace(cf["inspect_eh"]["start"],
                                   cf["inspect_eh"]["stop"],
                                   cf["inspect_eh"]["step"]))

# Initialise Bayesian Infrastructure
bay = ElHaddadBayes("dk_th", "ds_w", Y=cf["curve"]["Y"], name=cf["name"])
bay.load_log_likelihood(getattr(sklearn.metrics, cf["log_likelihood"]["function"]),
                        normalize=cf["log_likelihood"]["normalize"])

# Inspect likelihood (which coincides in this case) before MAP
v = BayesViewer("dk_th", cf["inspect_bayes"]["b1"], cf["inspect_bayes"]["n1"],
                "ds_w" , cf["inspect_bayes"]["b2"], cf["inspect_bayes"]["n2"],
                name=cf["name"])
v.config(save=cf["export"]["save"], folder=cf["export"]["folder"])
# v.contour("log_likelihood", bay, sd)

# Run MAP
bay.MAP(sd, cf["map_guess"])

# Get Optimal EH curve
opt = ElHaddadCurve(dk_th=bay.theta_hat[0], ds_w=bay.theta_hat[1], Y=cf["curve"]["Y"],
                    name=cf["name"] + "_Estimated")
opt.config(save=cf["export"]["save"], folder=cf["export"]["folder"])
opt.inspect(np.linspace(cf["inspect_eh"]["start"],
                        cf["inspect_eh"]["stop"],
                        cf["inspect_eh"]["step"]),
                        scale=cf["inspect_eh"]["scale"])

# View results
p = PreProViewer(cf["train"]["x1"], cf["train"]["x2"],
                 cf["inspect_eh"]["step"], cf["inspect_eh"]["scale"],
                 name=cf["name"])
p.config(save=cf["export"]["save"], folder=cf["export"]["folder"])
p.view(train_data=sd, curve=[eh, opt])
p.view(curve=[eh, opt])

print(bay.theta_hat)