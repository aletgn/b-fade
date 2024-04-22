from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

from bfade.abstract import AbstractCurve
from bfade.util import identity 
from bfade.dataset import SyntheticDataset
from bfade.elhaddad import ElHaddadCurve

import numpy as np

def el_haddad_distance(dk_th=5, ds_w=600, Y=0.65):
    eh = ElHaddadCurve(metrics = identity, dk_th=dk_th, ds_w=ds_w, Y=Y)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    signed_dist, x1_min, x2_min = eh.signed_distance_to_dataset(d)
    eh.inspect_signed_distance(np.linspace(1, 1000, 1000), x1_min, x2_min, signed_dist, d.X, scale="linear")


def el_haddad_distance_log(dk_th=5, ds_w=600, Y=0.65):
    eh = ElHaddadCurve(metrics = np.log10, dk_th=dk_th, ds_w=ds_w, Y=Y)
    
    d = SyntheticDataset()
    d.make_grid([1, 1000], [100,700], 20, 20, spacing="log")
    d.clear_points(eh, tol=1)
    d.make_classes(eh)
    d.inspect([1, 1000], [100,700], scale="log", curve=eh, x=np.linspace(1,1000,100))
    
    signed_dist, x1_min, x2_min = eh.signed_distance_to_dataset(d)
    eh.inspect_signed_distance(np.linspace(1, 1000, 1000), x1_min, x2_min, signed_dist, d.X, scale="log")


if __name__ == "__main__":
    # line_distance()
    # el_haddad_distance()
    # el_haddad_distance_log()
    pass
