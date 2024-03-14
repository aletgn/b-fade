#%% Import required modules and configure common parameters
from sys import path as syspath
from os import path as ospath

syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/b-fade/src'))

import numpy as np
import pandas as pd

from bfade.dataset import Dataset, SyntheticDataset
from bfade.elhaddad import ElHaddadDataset
from bfade.elhaddad import ElHaddadCurve

from bfade.viewers import PreProViewer

data_path = "/home/ale/Downloads/SyntheticEH.csv"

def istantiation_and_split():
    d = Dataset(X=np.array([[0,0],[1,1], [2,2]]), y=np.array([0,1,0]))
    d_tr, d_ts = d.partition()
    d = Dataset(X=np.array([[0,0],[1,1], [2,2]]), y=np.array([0,1,0]), test=np.array([1,1,0]))
    d_tr, d_ts = d.partition(method="user")
    print(d_tr.X, d_ts.X)

def read_split_inspect():
    d = Dataset(reader=pd.read_csv, path=data_path)
    eh = ElHaddadCurve(dk_th=3, ds_w=180, y=.73)
    d_tr, d_ts = d.partition("user")
    print(d_tr.X.shape, "\n", d_ts.X.shape)
    d_tr.inspect(xlim=[1,1000], ylim=[50,300], scale="log", curve=eh, x = np.linspace(1,1000,100))
    d_ts.inspect(xlim=[1,1000], ylim=[50,300], scale="log", curve=eh, x = np.linspace(1,1000,100))

def read_split_inspect_curve():
    eh = ElHaddadCurve(dk_th=3, ds_w=180, y=.73)
    d = ElHaddadDataset(reader=pd.read_csv, path=data_path)
    d.pre_process()
    d_tr, d_ts = d.partition(method="random")
    # d.inspect([1,1000], [1,1000], scale="log", curve=eh, x = np.linspace(1,1000,100))
    # d_tr.inspect([1,1000], [1,1000], scale="log", curve=eh, x = np.linspace(1,1000,100))
    # d_ts.inspect([1,1000], [1,1000], scale="log", curve=eh, x = np.linspace(1,1000,100))

    p = PreProViewer([10,1000], [50,300], scale="log")
    p.view(train_data = d_tr, test_data=d_ts)

def noisy_dataset():
    eh = ElHaddadCurve(dk_th=3, ds_w=180, y=.73)
    sd = SyntheticDataset()
    sd.make_grid([1,1000], [50,300], 20, 20, spacing="log")
    # sd.make_tube(eh, [1,1000], 50, 0.1, -0.1, 5, spacing="log")
    # sd.clear_points(eh, tol=20)
    sd.make_classes(eh)
    sd.add_noise(10,10)
    # sd.inspect(xlim=[1,1000], ylim=[50,300], scale="log")
    d_tr, d_ts = sd.partition(method="random")
    p = PreProViewer([10,1000], [50,300], scale="log")
    p.view(train_data=d_tr, test_data=d_ts)

    # print(sd.aux)

if __name__ == "__main__":
    # istantiation_and_split()
    # read_split_inspect()
    read_split_inspect_curve()
    noisy_dataset()
    pass