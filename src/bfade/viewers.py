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


class PreProViewer():
    
    def __init__(self, x_edges=[1,1000], y_edges=[100,700], n=1000, scale="linear", *deterministic):
        
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.x_scale = scale
        self.y_scale = scale
        self.n = n
        self.deterministic = deterministic
        
        if scale == "log":
            self.x = np.logspace(np.log10(x_edges[0]), np.log10(x_edges[1]), n)
        else:
            self.x = np.linspace(x_edges[0], x_edges[1], n)
    
    def view(self, **kwargs):
        self.fig, self.ax = plt.subplots(dpi=300)
        self.sr = None
        self.ss = None
        
        try:
            confidence = kwargs.pop("confidence")
        except:
            pass
        
        try:
            curve = kwargs.pop("ref_curve")
        except:
            pass
        
        try:
            post_samples = kwargs.pop("post_samples")
        except:
            pass
        
        try:
            det = [kwargs.pop(d) for d in self.deterministic]
            det_pars = dict(zip(self.deterministic, det))
        except KeyError:
            pass
        
        try:
            data = kwargs.pop("data")
        except KeyError:
            pass
        
        try:
            post_op = kwargs.pop("post_op")
        except KeyError:
            pass
        
        for k in kwargs:
            if k == "train_data":
                y0 = np.where(kwargs[k].y==0)
                y1 = np.where(kwargs[k].y==1)
                
                try:
                    c0=kwargs[k].aux[y0]
                    c1=kwargs[k].aux[y1]
                    vmin=kwargs[k].aux_min
                    vmax=kwargs[k].aux_max
                except:
                    c0 = [0]*len(y0[0])
                    c1 = [1]*len(y1[0])
                    vmin = 0
                    vmax = 1
                
                self.sr = self.ax.scatter(kwargs[k].X[y0, 0], kwargs[k].X[y0, 1],
                                          marker='o',
                                          c=c0, vmin=vmin, vmax=vmax,
                                          cmap='RdYlBu_r',
                                          edgecolor='k',
                                          s=50,
                                          # label='Runout', zorder=10
                                          )

                self.ax.scatter(kwargs[k].X[y1, 0], kwargs[k].X[y1, 1],
                                marker='X',
                                c=c1, vmin=vmin, vmax=vmax,
                                cmap='RdYlBu_r',
                                edgecolor='k',
                                s=50,
                                # label='Runout', zorder=10
                                )


            elif k == "test_data":
                y0 = np.where(kwargs[k].y==0)
                y1 = np.where(kwargs[k].y==1)
                
                try:
                    c0=kwargs[k].aux[y0]
                    c1=kwargs[k].aux[y1]
                    vmin=kwargs[k].aux_min
                    vmax=kwargs[k].aux_max
                except:
                    c0 = [0]*len(y0[0])
                    c1 = [1]*len(y1[0])
                    vmin = 0
                    vmax = 1

                y0 = np.where(kwargs[k].y==0)
                y1 = np.where(kwargs[k].y==1)
                self.ss = self.ax.scatter(kwargs[k].X[y0,0], kwargs[k].X[y0,1],
                                          marker='s',
                                          c=c0, vmin=vmin, vmax=vmax,
                                          cmap='RdYlBu_r',
                                          edgecolor='k',
                                          s=50,
                                          # label='Runout', zorder=10
                                          )
                
                self.ax.scatter(kwargs[k].X[y1,0], kwargs[k].X[y1,1],
                                          marker='P',
                                          c=c1, vmin=vmin, vmax=vmax,
                                          cmap='RdYlBu_r',
                                          edgecolor='k',
                                          s=50,
                                          # label='Runout', zorder=10
                                          )
            
            elif k == "curve":
                for c in kwargs[k]:
                    self.ax.plot(self.x, c.equation(self.x))
            
            elif k == "prediction_interval":
                mean, pred = kwargs[k].prediction_interval(self.x_edges, self.n, self.x_scale, curve, confidence, **det_pars)
                self.ax.plot(self.x, mean, "k")
                self.ax.plot(self.x, mean - pred, "k")
                self.ax.plot(self.x, mean + pred, "k")
            
            elif k == "predictive_posterior":
                predictions = post_op(kwargs[k].predictive_posterior(post_samples, data), axis=0)
                
                pp = self.ax.tricontourf(data.X[:,0], data.X[:,1], predictions,
                                         cmap='RdBu_r',
                                         levels=np.linspace(predictions.min(),
                                                            predictions.max()+1e-15, 21),
                                         antialiased='False')
        
                cbar = self.fig.colorbar(pp, ax=self.ax, orientation="vertical",
                                          pad=0.03, format="%.2f",
                                          # ticks=list(np.linspace(0, 1+1e-15, 21)),
                                          ticks = list(np.linspace(predictions.min(), predictions.max(), 11)),
                                          # label=self.translator[sel]
                                          )
                cbar.ax.tick_params(direction='in', top=1, size=2.5)
            
            else:
                raise KeyError
                
        self.ax.set_xscale(self.x_scale)
        self.ax.set_yscale(self.y_scale)
        self.ax.set_xlim(self.x_edges)
        self.ax.set_ylim(self.y_edges)
        self.ax.tick_params(direction="in", which='both', right=1, top=1)
            
