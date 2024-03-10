from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit

from sklearn.model_selection import train_test_split as tts

from bfade.abstract import AbstractBayes, AbstractCurve, AbstractDataset
from bfade.util import sif_equiv, inv_sif_range, sif_range
from bfade.util import MissingInputException, YieldException, logger_factory

_log = logger_factory(name=__name__, level="DEBUG")

class ElHaddadCurve(AbstractCurve):
    
    def __init__(self, **pars):    
        super().__init__(**pars)
    
    def equation(self, X: np.ndarray) -> np.ndarray:
        """
        Concrete representation of Evaluate El-Haddad curve over a given :math:`\sqrt{\\text{area}}` range.

        .. math::
            \Delta\sigma = \Delta\sigma_w\sqrt{{\sqrt{\\text{area}_0}}\
                                            \over{\sqrt{\\text{area}_0} \
                                                    + \sqrt{\\text{area}}}}

        where

        .. math::
            \sqrt{\\text{area}_0} = {1 \over \pi} \\bigg({{\Delta K_{th}}
                                            \over {Y \Delta \sigma_{w}}}\\bigg)^2
        
        Parameters
        ----------
            X : np.ndarray
                range of sqrt_area

        Returns
        -------
        np.ndarray
            Evaluated El Haddad curve along the given sqrt_area values.
            
        """
        self.sqrt_a0 = inv_sif_range(self.dk_th*1000, self.ds_w, self.y)
        return self.ds_w * ((self.sqrt_a0/(X+self.sqrt_a0))**0.5)


class ElHaddadDataset(AbstractDataset):

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def pre_process(self, **kwargs):
        """
        Pre-process the dataset:

             - set 'Y'

             - convert sqrt_area using the SIF equivalence

             - compute SIF

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Y_ref to specify the reference value for Y.

        Raises
        ------
        MissingInputException
            Raised if 'Y' is neither unique in the dataset nor provided\
            as a keyword argument.

        """
        _log.debug(f"{self.__class__.__name__}.{self.pre_process.__name__}")
        try:
            self.Y = kwargs.pop("Y_ref")
            _log.warning(f"Y_ref user-provided = {self.Y:.2f}")
        except KeyError:
            _log.warning(f"Y_ref not user-provided")
            _log.warning("Verify uniqueness of Y")
            if len(set(self.data.Y)) == 1:
                self.Y = list(set(self.data.Y))[0]
                _log.debug(f"Y is unique = {self.Y:.2f}")              
            else:
                _log.error(f"Y is not unique")
                _log.debug(f"Values found: {set(self.data.Y)}")
                raise MissingInputException("Y_ref is neither unique nor provided")

        _log.info("Update dataframe")
        self.data.rename(columns={"Y": "Y_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("Y_")+1, "Y", self.Y)

        _log.warning(f"Convert sqrt_area by {self.Y:.2f}")
        self.data.rename(columns={"sqrt_area": "sqrt_area_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("sqrt_area_")+1, "sqrt_area",
                        sif_equiv(self.data.sqrt_area_, self.data.Y_, self.Y))

        _log.info("Compute SIF range")
        self.data.insert(list(self.data.columns).index("Y")+1, "dk",
                        sif_range(self.data.delta_sigma, self.data.Y, self.data.sqrt_area*1e-6))

        _log.debug(f"Calculate min max of delta_k for colour bars")
        self.aux = self.data["dk"].to_numpy()
        self.aux_min = self.aux.min()
        self.aux_max = self.aux.max()

        return ElHaddadDataset(**self.populate("data"))

    def partition(self, method: str = "random", test_size: float = 0.2, rnd_state: int = 0) -> None:
        """
        Split dataset into seen (training) and unseen (test) points.

        Parameters
        ----------
        method : string, optional
            the parameters controls how to partition (split) the dataset. \
                if "random", then use the built in function of \
                    sklearn train_test_split. Else, if "user", then split\
                        according to the column "split" in the dataset.
            The default is "random".
        test_size : float, optional
            test_size controls the fraction of unseen (test) data against\
                those considered for MAP. The default is 0.2. \
                    Accepted values from 0.0.
            to 1.0
        rnd_state : int, optional
            random state for splitting. The default is 0.

        Raises
        ------
        Exception
            if "method" is not included in the possible choices, then throw an
            exception.

        Returns
        -------
        None.

        """
        _log.debug(f"{self.__class__.__name__}.{self.partition.__name__}")
        
        _log.warning(f"Train/test split. Method: {method}")
        if method == "random":
            self.split_method = method
            self.train, self.test = tts(self.data, test_size=test_size,
                                    random_state=rnd_state, shuffle=True)
        elif method == "user":
            self.split_method = method
            self.train = self.data.query("test == 0")
            self.test = self.data.query("test == 1")
        else:
            raise Exception("split method incorrectly provided")

        return ElHaddadDataset(**self.populate("train")), ElHaddadDataset(**self.populate("test"))

    def populate(self, data: str) -> Dict[str, Any]:
        """
        Compose dataset making attributes X = [sqrt_area, delta_sigma], y = failed, and aux = dk.

        Parameters
        ----------
        data : string
            

        Raises
        ------
        Exception
            if "method" is not included in the possible choices, then throw an
            exception.

        Returns
        -------
        Dict[str, Any]
            The composed dataset to be loaded.

        """
        return {"X": getattr(self, data)[["sqrt_area", "delta_sigma"]].to_numpy(),
                "y": getattr(self, data)["failed"].to_numpy(),
                "aux": getattr(self, data)["dk"].to_numpy(),
                "aux_min": self.aux_min,
                "aux_max": self.aux_max,
                "Y": self.Y}
    
    def inspect(self, scale="linear"):
        """
        Visualize the synthetic dataset and optionally plot the underlying curve.
    
        Parameters
        ----------
        scale : str, optional
            Scale for both x and y axes. Options are "linear" (default) or "log".
    
        Returns
        -------
        None
            Displays a scatter plot of the synthetic dataset with an optional plot of the underlying curve.
        """
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(self.X[:,0], self.X[:,1], c=self.y, s=10)        
        plt.xscale(scale)
        plt.yscale(scale)
        plt.show()


class ElHaddadBayes(AbstractBayes):

    def __init__(self, *pars, **args):
        super().__init__(*pars, **args)

    def predictor(self, D, *P: Dict[str, float]):
        """
        Perform logistic prediction based on the given parameters and dataset.

        .. math::
            P[\mathbf{x}_i | \\theta] = {{1}\over{1+\exp [-\mathcal{H}(\mathbf{x}_i, \\theta)]}}

        where :math:`\\theta` is the vector of trainable parameters

        .. math::
            \\theta = [\Delta K_{th,lc}\ \Delta\sigma_w]

        and :math:`\mathbf{x}_i \in D` is a sample from the given dataset.

        :math:`\mathcal{H}(\mathbf{x}_i, \\theta)` is the signed distance of the sample
        to the El Haddad curve of parameters :math:`\\theta`.

        The position of the training points wrt the target curve are computed over\
        the log-log plane.

        Parameters
        ----------
        D : ElHaddadDataset
        
        P : Dict[str, float]
            Dictionary of the trainable parameters

        Returns
        -------
        numpy.ndarray
            An array containing the logistic predictions.

        """
        eh = ElHaddadCurve(metrics=np.log10, dk_th=P[0], ds_w=P[1], y=0.65)
        signed_distance, _, _ = eh.signed_distance_to_dataset(D)
        return expit(signed_distance)
