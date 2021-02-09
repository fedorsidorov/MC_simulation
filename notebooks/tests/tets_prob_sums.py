import importlib

from _outdated.MC_classes_cm import Structure

from numpy import random

import arrays
import constants as const
import grid as grid
import indexes as indxs
from functions import MC_functions as utils

arrays = importlib.reload(arrays)
const = importlib.reload(const)
grid = importlib.reload(grid)
utils = importlib.reload(utils)
indxs = importlib.reload(indxs)

# %%
structure = Structure(100e-7)
now_list = structure.IMFP_norm

# %%
PMMA_arr = now_list[0]
for i, line in enumerate(PMMA_arr):
    el = random.choice([0, 1, 2, 3, 4, 5], p=line)

# %%
Si_arr = now_list[1]
for i, line in enumerate(Si_arr[indxs.Si_E_cut_ind:]):
    el = random.choice([0, 1, 2, 3, 4, 5, 6], p=line)


