import numpy as np
from scipy.special import expit
from bfade.abstract import AbstractBayes, AbstractCurve
from bfade.fracture import inv_sif_range

class ElHaddadCurve(AbstractCurve):
    
    def __init__(self, **pars):    
        super().__init__(**pars)
    
    def equation(self, X):
        self.sqrt_a0 = inv_sif_range(self.dk_th*1000, self.ds_w, self.y)
        return self.ds_w * ((self.sqrt_a0/(X+self.sqrt_a0))**0.5)
    
    def load_metrics(self, metrics):
        self.metrics = metrics


class ElHaddadBayes(AbstractBayes):

    def __init__(self, *pars, **args):
        super().__init__(*pars, **args)

    def predictor(self, D, *P):
        eh = ElHaddadCurve(metrics=np.log10, dk_th=P[0], ds_w=P[1], y=0.65)
        signed_distance, _, _ = eh.signed_distance_to_dataset(D.X)
        return expit(signed_distance)
