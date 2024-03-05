from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

from bfade.curve import AbstractCurve
from bfade.util import identity 

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
    
if __name__ == "__main__":
    istantiation()
    inspection()
