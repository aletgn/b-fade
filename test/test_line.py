from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

from bfade.curve import AbstractCurve
from bfade.util import identity 
from bfade.datagen import SyntheticDataset

import numpy as np

class Line(AbstractCurve):
    
    def __init__(self, **pars):
        super().__init__(**pars)
    
    def equation(self, X):
        return self.m*X + self.q
    
def istantiation(m = 1, q = 0):
    l = Line(m = m, q = q)
    l.load_metrics(identity)
    print(l)

def inspection(m = 1, q = 0):
    l = Line(m = m, q = q)
    l.inspect(np.linspace(-5, 5, 100))
    print(l)
    
def grid(m = 1, q = 0):
    l = Line(m = m, q = q)
    d = SyntheticDataset(l)
    d.make_grid([-5,5],[-5,5], 10, 10)
    d.inspect()
    d.make_classes()
    d.inspect()
    d.add_noise(0.1, 0.5)
    d.inspect()

def tube(m = 1, q = 0):
    l = Line(m = m, q = q)
    d = SyntheticDataset(l)
    d.make_tube([-5,5], up=3, down=-3, step=4)
    d.inspect()
    d.make_classes()
    d.inspect(np.linspace(-10,10,100))

if __name__ == "__main__":
    # istantiation()
    # inspection()
    # grid()
    # tube()
    pass
