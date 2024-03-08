import numpy as np
from scipy.special import expit

from sklearn.model_selection import train_test_split as tts

from bfade.abstract import AbstractBayes, AbstractCurve, AbstractDataset
from bfade.util import sif_equiv, inv_sif_range, sif_range
from bfade.util import MissingInputException

class ElHaddadCurve(AbstractCurve):
    
    def __init__(self, **pars):    
        super().__init__(**pars)
    
    def equation(self, X):
        self.sqrt_a0 = inv_sif_range(self.dk_th*1000, self.ds_w, self.y)
        return self.ds_w * ((self.sqrt_a0/(X+self.sqrt_a0))**0.5)
    
    def load_metrics(self, metrics):
        self.metrics = metrics


class ElHaddadDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pre_process(self, **kwargs):
        """
        Pre-process the dataset:

             - set 'Y'

             - convert sqrt_area using the SIF equivalence

             - compute SIF dk

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
        # _log.debug(f"{self.__class__.__name__}.{self.pre_process.__name__}")
        try:
            self.y_ref = kwargs.pop("Y_ref")
            # _log.warning(f"y_ref user-provided = {self.y_ref:.2f}")
        except KeyError:
            # _log.warning(f"y_ref not user-provided")
            # _log.warning("Verify uniqueness of y")
            if len(set(self.data.Y)) == 1:
                self.Y = list(set(self.data.Y))[0]
                # _log.debug(f"y_ref is unique = {self.y_ref:.2f}")              
            else:
                # _log.error(f"y is not unique")
                # _log.debug(f"Values found: {set(self.data.y)}")
                raise MissingInputException("y_ref is neither unique nor provided")

        # _log.info("Update dataframe")
        self.data.rename(columns={"Y": "Y_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("Y_")+1, "Y", self.Y)

        # _log.warning(f"Convert sqrt_area by {self.y_ref:.2f}")
        self.data.rename(columns={"sqrt_area": "sqrt_area_"}, inplace=True)
        self.data.insert(list(self.data.columns).index("sqrt_area_")+1, "sqrt_area",
                        sif_equiv(self.data.sqrt_area_, self.data.Y_, self.Y))

        # _log.info("Compute SIF range")
        self.data.insert(list(self.data.columns).index("Y")+1, "dk",
                        sif_range(self.data.delta_sigma, self.data.Y, self.data.sqrt_area*1e-6))

        # _log.debug(f"Calculate min max of delta_k for colour bars")
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
        # _log.debug(f"{self.__class__.__name__}.{self.partition.__name__}")
        # if self.scaler_delta_k or self.scaler_delta_sigma or self.scaler_sqrt_area:
            # raise YieldException("Partitioning must be done before defining scalers")
        # _log.warning(f"Train/test split. Method: {method}")
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

    def populate(self, data):
        return {"X": getattr(self, data)[["sqrt_area", "delta_sigma"]].to_numpy(),
                "y": getattr(self, data)["failed"].to_numpy(),
                "aux": self.aux,
                "aux_min": self.aux_min,
                "aux_max": self.aux_max,
                "Y": self.Y}


class ElHaddadBayes(AbstractBayes):

    def __init__(self, *pars, **args):
        super().__init__(*pars, **args)

    def predictor(self, D, *P):
        eh = ElHaddadCurve(metrics=np.log10, dk_th=P[0], ds_w=P[1], y=0.65)
        signed_distance, _, _ = eh.signed_distance_to_dataset(D)
        return expit(signed_distance)
