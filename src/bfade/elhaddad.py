from bfade.abstract import AbstractCurve
from bfade.fracture import inv_sif_range

class ElHaddadCurve(AbstractCurve):
    
    def __init__(self, **pars):    
        super().__init__(**pars)
    
    def equation(self, X):
        self.sqrt_a0 = inv_sif_range(self.dk_th*1000, self.ds_w, self.y)
        return self.ds_w * ((self.sqrt_a0/(X+self.sqrt_a0))**0.5)
    
    def load_metrics(self, metrics):
        self.metrics = metrics
