from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from bfade.util import grid_factory, logger_factory,  state_modifier, printer
from bfade.abstract import AbstractMAPViewer

_log = logger_factory(name=__name__, level="DEBUG")

class BayesViewer(AbstractMAPViewer):
    
    def __init__(self, p1: str, b1: list, n1: int, p2: str, b2: list, n2: int, spacing: str = "lin", **kwargs: Dict[str, Any]) -> None:
        super().__init__(p1, b1, n1, p2, b2, n2, spacing, **kwargs)
        self.config()
        self.config_contour()

    @printer
    def contour(self, element="log_prior", bayes=None, dataset=None):
        """
        Create a contour plot for the specified element.

        Parameters
        ----------
        element : str, optional
            The element for which the contour plot is generated. Default is "log_prior".
        bayes : ElHaddadBayes, optional
            An instance of the Bayesian class. Default is default None.
        dataset : AbstractDataset
            The trainin dataset. Default is None.

        Returns
        -------
        None
        """
        _log.debug(f"{self.__class__.__name__}.{self.contour.__name__}. Contour: {element:s}")
        fig, ax = plt.subplots(dpi=300)
        
        if element == "log_prior":
            el_cnt = np.array([getattr(bayes, element)(pp1, pp2) for pp1,pp2, in 
                               zip(getattr(self, self.p1), getattr(self, self.p2))])
        else:
            el_cnt = np.array([getattr(bayes, element)(dataset, pp1, pp2) for pp1,pp2, in 
                               zip(getattr(self, self.p1), getattr(self, self.p2))])
        
        cnt =  ax.tricontour(getattr(self, self.p1), getattr(self, self.p2), el_cnt,
                             levels=np.linspace(el_cnt.min(), el_cnt.max(), self.levels),
                             cmap=self.cmap)
        

        cbar = plt.gcf().colorbar(cnt, ax=ax,
                                  orientation="vertical",
                                  pad=0.1,
                                  format="%.3f",
                                  ticks=np.linspace(el_cnt.min(), el_cnt.max(), self.clevels),
                                  label=element,
                                  alpha=0.65)
        
        ax.set_xlabel(self.p1)
        ax.set_ylabel(self.p2)
        ax.tick_params(direction='in', top=1, right =1)
        cbar.ax.tick_params(direction='in', top=1, size=2.5)
        
        return fig, self.name + "_bay_" + element


class LaplacePosteriorViewer(AbstractMAPViewer):
    
    def __init__(self, p1: str, c1: float, n1: int, p2: str, c2: float, n2: int, bayes, **kwargs: Dict[str, Any]) -> None:
        """
        Initialize LaplacePosteriorViewer.

        Parameters
        ----------
        p1 : str
            Name of the first parameter.
        c1 : float
            Coverage factore for the first parameter.
        n1 : int
            Number of grid points for the first parameter.
        p2 : str
            Name of the second parameter.
        c2 : float
            Coverage factore for  the second parameter.
        n2 : int
            Number of grid points for the second parameter.
        bayes : AbstractBayes
            An instance of AbstractBayes.

        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.__init__.__name__}")
        self.c1 = c1
        self.c2 = c2
        
        idx_1 = bayes.pars.index(p1)
        idx_2 = bayes.pars.index(p2)
        b1 = np.array([-c1, c1])*(bayes.ihess[idx_1][idx_1]**0.5) + bayes.theta_hat[idx_1]
        b2 = np.array([-c2, c2])*(bayes.ihess[idx_2][idx_2]**0.5) + bayes.theta_hat[idx_2]
        
        super().__init__(p1, b1, n1, p2, b2, n2, spacing="lin", **kwargs)
        self.config()
        self.config_contour()
    
    @printer
    def contour(self, bayes) -> None:
        """
        Plot joint posterior distribution.

        """
        _log.debug(f"{self.__class__.__name__}.{self.contour.__name__} -- joint poterior")
        fig, ax = plt.subplots(dpi=300)
        el_cnt = np.array([bayes.joint.pdf([pp1, pp2]) for pp1, pp2 
                                      in zip(getattr(self, self.p1), getattr(self, self.p2))])
        
        cnt =  ax.tricontour(getattr(self, self.p1), getattr(self, self.p2), el_cnt,
                             levels=np.linspace(el_cnt.min(), el_cnt.max(), 21))
        
        cbar = plt.gcf().colorbar(cnt, ax=ax,
                                  orientation="vertical",
                                  pad=0.1,
                                  format="%.3f",
                                  label="joint posterior",
                                  alpha=0.65)
        ax.tick_params(direction='in', top=1, right =1)
        cbar.ax.tick_params(direction='in', top=1, size=2.5)
        
        return fig, self.name + "_laplace_joint"

    @printer
    def marginals(self, p: str, bayes) -> None:
        """
        Plot marginal posterior distribution.

        Parameters
        ----------
        p : str
            Name of the parameter to be inspected.
        bayes : AbstractBayes
            Instance of AbstractBayes to query.

        Returns
        -------
        None
        
        """
        _log.debug(f"{self.__class__.__name__}.{self.marginals.__name__}")
        fig, ax = plt.subplots(dpi=300)
        ax.plot(np.sort(getattr(self, p)),
                getattr(bayes, "marginal_" + p).pdf(np.sort(getattr(self, p))), "k")
        ax.set_xlabel(p)
        ax.set_ylabel("marginal posterior")
        ax.set_title(f"mean = {getattr(bayes, 'marginal_' + p).mean():.2f}" + \
                     f"-- st. dev. = {getattr(bayes, 'marginal_' + p).std():.2f}")

        ax.tick_params(direction='in', top=1, right =1)
        
        return fig, self.name + "_lap_marginal_" + p


class PreProViewer():
    
    def __init__(self, x_edges=[1,1000], y_edges=[100,700], n=1000, scale="linear",
                 **args: Dict[str, Any]) -> None:
        
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.x_scale = scale
        self.y_scale = scale
        self.n = n
        
        try:
            self.name = args.pop("name")
        except:
            self.name = "Untitled"
        
        self.det_pars = args

        _log.debug(f"{self.__class__.__name__}.{self.__init__.__name__} -- {self}")
        
        if scale == "log":
            self.x = np.logspace(np.log10(x_edges[0]), np.log10(x_edges[1]), n)
        else:
            self.x = np.linspace(x_edges[0], x_edges[1], n)
        
        self.config()
        self.config_canvas()

    def config(self, save: bool = False, folder: str = "./", fmt: str = "png", dpi: int = 300, 
               legend_config: Dict[str, Any] = None):
        _log.debug(f"{self.__class__.__name__}.{self.config.__name__}")
        self.save = save
        self.folder = folder
        self.fmt = fmt
        self.dpi = dpi
        self.legend_config = legend_config

    def config_canvas(self, xlabel: str = "x1", ylabel: str = "x2", cbarlabel: str = " ",
               class0: str = "0", class1: str = "1") -> None:
        
        _log.debug(f"{self.__class__.__name__}.{self.config_canvas.__name__}")
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cbarlabel = cbarlabel
        self.class0 = class0
        self.class1 = class1
       
    def add_colourbar(self, ref, vmin, vmax):
        """
        Add a colorbar to the El Haddad plot.

        Parameters
        ----------
        ref : matplotlib.image.AxesImage
            A reference to the image onto which the colorbar is drawn.

        Returns
        -------
        None

        """
        _log.debug(f"{self.__class__.__name__}.{self.add_colourbar.__name__}")
        cbar = self.fig.colorbar(ref, ax=self.ax, orientation="vertical",
                                  pad=0.05, format="%.1f",
                                  ticks=list(np.linspace(vmin, vmax, 11)),
                                  label=self.cbarlabel)
        
        cbar.ax.tick_params(direction='in', top=1, size=2.5)

    @printer
    def view(self, **kwargs):
        self.fig, self.ax = plt.subplots(dpi=self.dpi)
        self.sr = None
        self.ss = None
        self.state = self.name
               
        try:
            post_samples = kwargs.pop("post_samples")
        except:
            pass
        
        try:
            post_data = kwargs.pop("post_data")
        except KeyError:
            pass
        
        try:
            post_op = kwargs.pop("post_op")
        except KeyError:
            pass
        
        for k in kwargs:
            if k == "train_data":
                _log.info("Inspect training data")
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
                                          label=self.class0+" (train)", zorder=10
                                          )

                self.ax.scatter(kwargs[k].X[y1, 0], kwargs[k].X[y1, 1],
                                marker='X',
                                c=c1, vmin=vmin, vmax=vmax,
                                cmap='RdYlBu_r',
                                edgecolor='k',
                                s=50,
                                label=self.class1+" (train)", zorder=10
                                )
                if self.ss is None:
                    self.add_colourbar(self.sr, vmin, vmax)
                self.state = state_modifier(self.state, "test", "data", "train")
                _log.debug(f"State: {self.state}")

            elif k == "test_data":
                _log.info("Inspect test data")
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

                self.ss = self.ax.scatter(kwargs[k].X[y0,0], kwargs[k].X[y0,1],
                                          marker='s',
                                          c=c0, vmin=vmin, vmax=vmax,
                                          cmap='RdYlBu_r',
                                          edgecolor='k',
                                          s=50,
                                          label=self.class0+" (test)", zorder=10
                                          )
                
                self.ax.scatter(kwargs[k].X[y1,0], kwargs[k].X[y1,1],
                                          marker='P',
                                          c=c1, vmin=vmin, vmax=vmax,
                                          cmap='RdYlBu_r',
                                          edgecolor='k',
                                          s=50,
                                          label=self.class1+" (test)", zorder=10
                                          )
                if self.sr is None:
                    self.add_colourbar(self.ss, vmin, vmax)
                self.state = state_modifier(self.state, "train", "data", "test")
                _log.debug(f"State: {self.state}")

            elif k == "curve":
                _log.info("Inspect given curves")
                for c in kwargs[k]:
                    self.ax.plot(self.x, c.equation(self.x), label=c.name)
                    
                    self.state += "_" + c.name.replace(" ", "")
                    _log.debug(f"State: {self.state}")

            elif k == "prediction_interval":
                _log.info("Inspect prediction interval")
                mean, pred, _ = kwargs[k].prediction_interval(self.x_edges, self.n, self.x_scale, **self.det_pars)
                # self.ax.plot(self.x, mean, "k")
                self.ax.plot(self.x, mean - pred, "-.k", label=fr"Pred. band. (@{50 - kwargs[k].confidence/2}$\%$)")
                self.ax.plot(self.x, mean + pred, "--k", label=fr"Pred. band. (@{50 + kwargs[k].confidence/2}$\%$)")
                self.state += "_pi"
                _log.debug(f"State: {self.state}")
            
            elif k == "predictive_posterior":
                _log.info("Inspect predictive posterior")
                predictions = post_op(kwargs[k].predictive_posterior(post_samples, post_data), axis=0)
                
                pp = self.ax.tricontourf(post_data.X[:,0], post_data.X[:,1], predictions,
                                         cmap='RdBu_r',
                                         levels=np.linspace(predictions.min(), predictions.max()+1e-15, 21),
                                         antialiased='False')
        
                cbar = self.fig.colorbar(pp, ax=self.ax, orientation="vertical",
                                          pad=0.03, format="%.2f",
                                          ticks = list(np.linspace(predictions.min(), predictions.max(), 11)),
                                          label=post_op.__name__)
                cbar.ax.tick_params(direction='in', top=1, size=2.5)
                self.state += "_" + post_op.__name__
                _log.debug(f"State: {self.state}")
            
            else:
                raise KeyError
                
        self.ax.set_xscale(self.x_scale)
        self.ax.set_yscale(self.y_scale)
        self.ax.set_xlim(self.x_edges)
        self.ax.set_ylim(self.y_edges)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.tick_params(direction="in", which='both', right=1, top=1)
        
        try:
            legend = self.ax.legend(**self.legend_config)
            _log.debug(f"{__class__.__name__}.{self.view.__name__}. Custom legend config")
        except:
            _log.info(f"{__class__.__name__}.{self.view.__name__}. Legend Setting 'best'")
            legend = self.ax.legend(loc="best")
        
        return self.fig, self.state

    def __repr__(self):
        attributes_str = ',\n '.join(f'{key} = {value}' for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attributes_str})"
            
