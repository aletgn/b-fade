from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
from scipy.stats import norm

from bfade.datagen import SyntheticDataset
from bfade.elhaddad import ElHaddadCurve
from bfade.core import Bayes
from bfade.statistics import uniform

from sklearn.metrics import log_loss

def istantiation():
    b = Bayes()
    b.declare_parameters("dkt", "dsw")
    print(b)
    
def priors():
    b = Bayes()
    b.declare_parameters("dkt", "dsw")
    b.load_prior("dkt", uniform, unif_value=1)
    b.load_prior("dsw", norm, loc=0, scale=1)
    print(b)
    return b

def likelihood():
    b = Bayes()
    b.declare_parameters("dkt", "dsw")
    b.load_log_likelihood(log_loss)
    print(b)
    return b

def calc_prior():
    b = Bayes()
    b.declare_parameters("dkt", "dsw")
    b.load_prior("dkt", uniform, unif_value=1)
    b.load_prior("dsw", norm, loc=0, scale=1)
    p = b.log_prior([7,8])
    print(b)
    print(p)
    
def calc_likelihood():
    pass

def calc_posterior():
    pass

if __name__ == "__main__":
    # istantiation()
    # b = priors()
    # likelihood()
    calc_prior()