from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np

from bfade.datagen import SyntheticDataset
from bfade.elhaddad import ElHaddadCurve

def istantiation(dk_th=5, ds_w=600, y=0.65):
    eh = ElHaddadCurve(dk_th=dk_th, ds_w=ds_w, y=y)
    print(eh)

def inspection(dk_th=5, ds_w=600, y=0.65):
    eh = ElHaddadCurve(dk_th=dk_th, ds_w=ds_w, y=y)
    eh.inspect(np.linspace(1, 1000, 1000), scale="log")
    
def grid(dk_th=5, ds_w=600, y=0.65):
    eh = ElHaddadCurve(dk_th=dk_th, ds_w=ds_w, y=y)
    eh.inspect(np.linspace(1, 1000, 1000), scale="log")
    d = SyntheticDataset(eh)
    d.make_grid([1, 1000],[200, 700], 20, 20, spacing="log")
    d.inspect(scale="log")
    d.clear_points(tol=10)
    d.inspect(np.linspace(1,1000,1000), scale="log")
    d.make_classes()
    d.inspect(np.linspace(1,1000,1000), scale="log")

def tube(dk_th=5, ds_w=600, y=0.65):
    eh = ElHaddadCurve(dk_th=dk_th, ds_w=ds_w, y=y)
    d = SyntheticDataset(eh)
    d.make_tube([1,1000], up=0.5, down=-0.5, step=5, spacing="log")
    d.inspect(scale="log")
    d.make_classes()
    d.inspect(np.linspace(1,1000,1000), scale="log")
    print(eh)

if __name__ == "__main__":
    # istantiation()
    # inspection()
    # grid()
    # tube()
    pass
