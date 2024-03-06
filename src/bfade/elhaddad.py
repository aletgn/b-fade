from bfade.curve import AbstractCurve
from bfade.fracture import inv_sif_range

class ElHaddadCurve(AbstractCurve):
    
    def __init__(self, **pars):    
        super().__init__(**pars)
    
    def equation(self, X):
        self.sqrt_a0 = inv_sif_range(self.dk_th*1000, self.ds_w, self.y)
        return self.ds_w * ((self.sqrt_a0/(X+self.sqrt_a0))**0.5)
    
    def load_metrics(self, metrics):
        self.metrics = metrics
        
    # def load_scalers(self, scaler, X):
    #     self.sa_sc = StandardScaler().fit(X[:,0].reshape(-1,1))
    #     self.ds_sc = StandardScaler().fit(X[:,1].reshape(-1,1))
    #     self.dk_sc = StandardScaler().fit(sif_range(X[:,1], self.y, X[:,0]*1e-6).reshape(-1,1))