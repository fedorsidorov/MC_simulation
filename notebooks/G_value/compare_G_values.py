import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import constants as const
from mapping import mapping_harris as mapping
from functions import G_functions as Gf
from functions import MC_functions as mcf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
mcf = importlib.reload(mcf)
Gf = importlib.reload(Gf)

# %%
weights = np.load('notebooks/G_value/samples/weights.npy')

TT_sim_1 = np.load('notebooks/G_value/samples/TT_sim_1.npy')
TT_sim_2 = np.load('notebooks/G_value/samples/TT_sim_2.npy')
TT_sim_3 = np.load('notebooks/G_value/samples/TT_sim_3.npy')

plt.figure(dpi=300)
plt.plot(TT_sim_1, weights, label='1')
plt.plot(TT_sim_2, weights, label='2')
plt.plot(TT_sim_3, weights, label='3')

plt.legend()
plt.grid()
plt.show()

