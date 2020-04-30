import importlib

import matplotlib.pyplot as plt

import numpy as np
from numpy import random

import grid as grid
import constants as c
import arrays as a
import utilities as u

a = importlib.reload(a)
c = importlib.reload(c)
grid = importlib.reload(grid)
u = importlib.reload(u)

# %%
now_list = a.structure_IMFP_norm

# %%
PMMA_arr = now_list[0]
for i, line in enumerate(PMMA_arr):
    el = random.choice([0, 1, 2, 3, 4, 5], p=line)

# %%
Si_arr = now_list[1]
for i, line in enumerate(Si_arr[c.Si_MuElec_E_ind_plasmon:]):
    el = random.choice([0, 1, 2, 3, 4, 5, 6], p=line)


