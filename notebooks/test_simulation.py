# %%
import importlib

import numpy as np

import plot_data as p

p = importlib.reload(p)

DATA = np.load('data/e_DATA/DATA_0.npy')
p.plot_DATA(DATA, 500)
