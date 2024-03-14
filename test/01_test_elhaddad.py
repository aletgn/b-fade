from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np

from bfade.dataset import SyntheticDataset
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
    d = SyntheticDataset()
    d.make_grid([1, 1000],[200, 700], 20, 20, spacing="log")
    d.inspect([1, 1000],[200, 700], scale="log", curve=eh, x=np.linspace(1,1000, 1000))
    d.clear_points(eh, tol=10)
    d.inspect([1, 1000],[200, 700], scale="log", curve=eh, x=np.linspace(1,1000, 1000))
    d.make_classes(eh)
    d.inspect([1, 1000],[200, 700], scale="log", curve=eh, x=np.linspace(1,1000, 1000))

def tube(dk_th=5, ds_w=600, y=0.65):
    eh = ElHaddadCurve(dk_th=dk_th, ds_w=ds_w, y=y)
    d = SyntheticDataset()
    d.make_tube(eh, [1,1000], up=0.5, down=-0.5, step=5, spacing="log")
    d.inspect([1, 2000],[1, 2000], scale="log", curve=eh, x=np.linspace(1,1000, 1000))
    d.clear_points(eh, tol=10)
    d.make_classes(eh)
    d.inspect([1, 2000],[1, 2000], scale="log", curve=eh, x=np.linspace(1,1000, 1000))
    print(eh)

if __name__ == "__main__":
    # istantiation()
    # inspection()
    # grid()
    # tube()
    pass
