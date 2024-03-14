from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

from bfade.abstract import AbstractCurve
from bfade.util import identity 
from bfade.dataset import SyntheticDataset

import numpy as np

class Line(AbstractCurve):
    
    def __init__(self, **pars):
        super().__init__(**pars)
    
    def equation(self, X):
        return self.m*X + self.q
    
def istantiation(m = 1, q = 0):
    l = Line(m = m, q = q)
    print(l)

def inspection(m = 1, q = 0):
    l = Line(m = m, q = q)
    l.inspect(np.linspace(-5, 5, 100))
    print(l)
    
def grid(m = 1, q = 0):
    l = Line(m = m, q = q)
    d = SyntheticDataset()
    d.make_grid([-5, 5],[-5, 5], 10, 10)
    d.inspect([-5, 5],[-5, 5], scale="linear")
    d.make_classes(l)
    d.inspect([-5, 5],[-5, 5], scale="linear", curve=l, x=np.linspace(-5,5,100))
    d.add_noise(0.1, 0.5)
    d.inspect([-5, 5],[-5, 5], scale="linear")

def tube(m = 1, q = 0):
    l = Line(m = m, q = q)
    d = SyntheticDataset()
    d.make_tube(curve=l, x_bounds=[-5,5], n=50, up=3, down=-3, step=4)
    d.inspect([-5, 5],[-5, 5])
    d.make_classes(l)
    d.inspect([-5, 5],[-5, 5], scale="linear", curve=l, x=np.linspace(-5,5,100))

if __name__ == "__main__":
    # istantiation()
    # inspection()
    # grid()
    # tube()
    pass
