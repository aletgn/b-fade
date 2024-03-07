from bfade.util import grid_factory
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class AbstractMAPViewer:
    
    def __init__(self, p1, b1, n1, p2, b2, n2, spacing):
        
        self.pars = (p1, p2)
        self.p1 = p1
        self.p2 = p2
        self.n1 = n1
        self.n2 = n2
        self.spacing = spacing
        setattr(self, "bounds_" + p1, b1)
        setattr(self, "bounds_" + p2, b2)
        
        X1, X2 = grid_factory(getattr(self, "bounds_" + p1),
                              getattr(self, "bounds_" + p2),
                              self.n1, self.n2, spacing)
        setattr(self, p1, X1)
        setattr(self, p2, X2)
    
    @abstractmethod
    def contour(self):
        ...
    
    def config_contour(self):
        pass
    
    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"


class BayesViewer(AbstractMAPViewer):
    
    def __init__(self, p1, b1, n1, p2, b2, n2, spacing):
        super().__init__(p1, b1, n1, p2, b2, n2, spacing)
        
    def contour(self, element="log_prior", bayes=None, dataset=None):
        fig, ax = plt.subplots(dpi=300)
        
        if element == "log_prior":
            el_cnt = np.array([getattr(bayes, element)(pp1, pp2) for pp1,pp2, in 
                               zip(getattr(self, self.p1), getattr(self, self.p2))])
        else:
            el_cnt = np.array([getattr(bayes, element)(dataset, pp1, pp2) for pp1,pp2, in 
                               zip(getattr(self, self.p1), getattr(self, self.p2))])
        
        cnt =  ax.tricontour(getattr(self, self.p1), getattr(self, self.p2), el_cnt,
                             levels=np.linspace(el_cnt.min(), el_cnt.max(), 21))
        
        cbar = plt.gcf().colorbar(cnt, ax=ax,
                                  orientation="vertical",
                                  pad=0.1,
                                  format="%.1f",
                                  label=element,
                                  alpha=0.65)


class LaplacePosteriorViewer(AbstractMAPViewer):
    
    def __init__(self, p1, c1, n1, p2, c2, n2, bayes):
        
        self.c1 = c1
        self.c2 = c2
        
        idx_1 = bayes.pars.index(p1)
        idx_2 = bayes.pars.index(p2)
        b1 = np.array([-c1, c1])*(bayes.ihess[idx_1][idx_1]**0.5) + bayes.theta_hat[idx_1]
        b2 = np.array([-c2, c2])*(bayes.ihess[idx_2][idx_2]**0.5) + bayes.theta_hat[idx_2]
        
        super().__init__(p1, b1, n1, p2, b2, n2, spacing="lin")

    def config_marginals():
        pass
    
    def contour(self, bayes):
        fig, ax = plt.subplots(dpi=300)
        
        el_cnt = np.array([bayes.joint.pdf([pp1, pp2]) for pp1, pp2 
                                      in zip(getattr(self, self.p1), getattr(self, self.p2))])
        
        cnt =  ax.tricontour(getattr(self, self.p1), getattr(self, self.p2), el_cnt,
                             levels=np.linspace(el_cnt.min(), el_cnt.max(), 21))
        
        cbar = plt.gcf().colorbar(cnt, ax=ax,
                                  orientation="vertical",
                                  pad=0.1,
                                  format="%.3f",
                                  label="posterior",
                                  alpha=0.65)
    
    def marginals(self, bayes):
        for p in bayes.pars:
            fig, ax = plt.subplots(dpi=300)
            ax.plot(np.sort(getattr(self, p)),
                    getattr(bayes, "marginal_" + p).pdf(np.sort(getattr(self, p))))
