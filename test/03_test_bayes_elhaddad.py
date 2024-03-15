from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bfade.dataset import SyntheticDataset
from bfade.elhaddad import ElHaddadCurve
from bfade.elhaddad import ElHaddadBayes
from bfade.statistics import uniform
from bfade.util import grid_factory

from sklearn.metrics import log_loss

def contour(x, y, z):
    fig, ax = plt.subplots(dpi=300)
    cnt = ax.tricontour(x, y, z,
                        levels=np.linspace(z.min(), z.max(), 21)
                        )
    plt.gcf().colorbar(cnt, ax=ax,
                       orientation="vertical",
                       pad=0.1,
                       format="%.1f",
                       # label=el2latex[element],
                       alpha=0.65)
    ax.tick_params(direction='in', top=1, right=1)
    plt.show()

def istantiation():
    b = ElHaddadBayes("dkt", "dsw")
    print(b)
    
def priors():
    b = ElHaddadBayes("dkt", "dsw")
    b.load_prior("dkt", uniform, unif_value=1)
    b.load_prior("dsw", norm, loc=0, scale=1)
    print(b)
    return b

def likelihood():
    b = ElHaddadBayes("dkt", "dsw")
    b.load_log_likelihood(log_loss)
    print(b)
    return b

def calc_prior():
    b = ElHaddadBayes("dkt", "dsw")
    b.load_prior("dkt", uniform, unif_value=1)
    b.load_prior("dsw", norm, loc=0, scale=1)
    p = b.log_prior(7,8)
    print(b)
    print(p)
    
def calc_likelihood():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 5, 5, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_log_likelihood(log_loss, normalize=True)
    
    l = b.log_likelihood(d, 5, 600)
    
    print(b)
    print(l)
    
def display_log_likelihood():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    # d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_log_likelihood(log_loss, normalize=True)
    
    dk_th, ds_w = grid_factory([3, 7], [400, 800], 10, 10, spacing="lin")
    llh = np.array([b.log_likelihood(d, k, w) for k,w in zip(dk_th, ds_w)])
    
    contour(dk_th, ds_w, llh)
    
def display_log_prior():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    # d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    
    dk_th, ds_w = grid_factory([3, 7], [400, 800], 25, 25, spacing="lin")
    lpr = np.array([b.log_prior(k, w) for k,w in zip(dk_th, ds_w)])
    
    contour(dk_th, ds_w, lpr)

def display_log_posterior():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    # d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    b.load_log_likelihood(log_loss, normalize=True)
    
    dk_th, ds_w = grid_factory([3, 7], [400, 800], 10, 10, spacing="lin")
    lpo = np.array([b.log_posterior(d, k, w) for k,w in zip(dk_th, ds_w)])
    
    contour(dk_th, ds_w, lpo)
    
def display_bayes_tube():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_tube(eh, [1, 1000], n=50, up=0.1, down=-0.1, step=5, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    b.load_log_likelihood(log_loss, normalize=True)
    
    dk_th, ds_w = grid_factory([3, 7], [400, 800], 10, 10, spacing="lin")
    
    llh = np.array([b.log_likelihood(d, k, w) for k,w in zip(dk_th, ds_w)])
    lpr = np.array([b.log_prior(k, w) for k,w in zip(dk_th, ds_w)])
    lpo = np.array([b.log_posterior(d, k, w) for k,w in zip(dk_th, ds_w)])
    
    contour(dk_th, ds_w, llh)
    contour(dk_th, ds_w, lpr)
    contour(dk_th, ds_w, lpo)
    
def run_map():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    # d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_prior("dk_th", norm, loc=5, scale=1)
    b.load_prior("ds_w", norm, loc=600, scale=50)
    b.load_log_likelihood(log_loss, normalize=True)
    np.seterr(divide='ignore', 
              invalid='ignore')
    b.MAP(d)
    
    print(b)
    
def likelihood_args():
    eh = ElHaddadCurve(metrics = np.log10, dk_th=5, ds_w=600, Y=0.65)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    # d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    b = ElHaddadBayes("dk_th", "ds_w", Y=0.65)
    b.load_log_likelihood(log_loss, normalize=True, eps="auto")
    
    # dk_th, ds_w = grid_factory([3, 7], [400, 800], 5, 5, spacing="lin")
    # llh = np.array([b.log_likelihood(d, k, w) for k,w in zip(dk_th, ds_w)])
    
    # contour(dk_th, ds_w, llh)
    
    print(b)
    
    
if __name__ == "__main__":
    # istantiation()
    # b = priors()
    # calc_prior()
    # calc_likelihood()
    # display_log_likelihood()
    # display_log_prior()
    # display_log_posterior()
    # display_bayes_tube()
    # run_map()
    likelihood_args()
    pass
