# %%
import importlib

import numpy as np

from SimClasses import plot_data as p

p = importlib.reload(p)

np.set_printoptions(precision=4)
DATA = np.load('data/e_DATA/DATA_0.npy')
p.plot_DATA(DATA, 500)
