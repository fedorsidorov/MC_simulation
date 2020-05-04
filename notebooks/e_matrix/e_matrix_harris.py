# %%
import importlib

import numpy as np
import indexes as ind
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import array_functions as af

ind = importlib.reload(ind)
pf = importlib.reload(pf)
emf = importlib.reload(emf)
af = importlib.reload(af)

DATA = np.load('data/e_DATA/DATA_test.npy')
DATA_5 = emf.get_e_id_DATA(DATA, 5)
# DATA = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA/DATA_test_1.npy')
# emf.add_xy_shift(DATA, 0, 1000, 0)
# emf.add_xy_shift(DATA, 1, -1000, 0)
# emf.add_xy_shift(DATA, 2, 2000, 0)
# emf.add_xy_shift(DATA, 3, -2000, 0)
af.snake_array(DATA, ind.DATA_x_ind, ind.DATA_y_ind, ind.DATA_z_ind, [-50, -50, -np.inf], [50, 50, np.inf])
pf.plot_DATA(DATA_5, d_PMMA=500, E_cut=1, proj='xz')
