import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import importlib

import grid

grid = importlib.reload(grid)


# %%
def get_OSC(hw, E, A, w):
    return A * w * hw / ((hw**2 - E**2)**2 + (hw * w)**2)


def get_OSC_edge(hw, E, A, w):
    OLF = get_OSC(hw, E, A, w)
    OLF[np.where(hw < E)] = 0
    return OLF


def get_OLF_E(hw, A1, w1, A2, w2, A3, w3, A4, w4):
    E1, E2, E3, E4 = 16.7, 20, 102, 151.55

    OLF = get_OSC(hw, E1, A1, w1)

    OLF += get_OSC_edge(hw, E2, A2, w2)
    OLF += get_OSC_edge(hw, E3, A3, w3)
    OLF += get_OSC_edge(hw, E4, A4, w4)

    return OLF


def get_OLF(hw, E1, A1, w1, E2, A2, w2, E3, A3, w3, E4, A4, w4):
    OLF = get_OSC(hw, E1, A1, w1)

    OLF += get_OSC_edge(hw, E2, A2, w2)
    OLF += get_OSC_edge(hw, E3, A3, w3)
    OLF += get_OSC_edge(hw, E4, A4, w4)

    return OLF


def get_OLF_E_log(hw_log, A1, w1, A2, w2, A3, w3, A4, w4):
    hw = np.exp(hw_log)
    E1, E2, E3, E4 = 16.7, 20, 102, 151.55

    OLF = get_OSC(hw, E1, A1, w1)

    OLF += get_OSC_edge(hw, E2, A2, w2)
    OLF += get_OSC_edge(hw, E3, A3, w3)
    OLF += get_OSC_edge(hw, E4, A4, w4)

    return np.log(OLF)


def get_OLF_log(hw_log, E1, A1, w1, E2, A2, w2, E3, A3, w3, E4, A4, w4):
    hw = np.exp(hw_log)

    OLF = get_OSC(hw, E1, A1, w1)

    OLF += get_OSC_edge(hw, E2, A2, w2)
    OLF += get_OSC_edge(hw, E3, A3, w3)
    OLF += get_OSC_edge(hw, E4, A4, w4)

    return np.log(OLF)


# %%
EE = np.load('notebooks/Akkerman_Si_5osc/OLF_Palik+Fano/EE_Palik+Fano.npy')[1179:4217]
OLF = np.load('notebooks/Akkerman_Si_5osc/OLF_Palik+Fano/OLF_Palik+Fano.npy')[1179:4217]

EE_log = np.log(EE)
OLF_log = np.log(OLF)

plt.figure(dpi=300)
plt.loglog(EE, OLF, '.')
# plt.plot(EE_log, OLF_log, '.')
plt.grid()
# plt.show()

p0 = [
    16.7, 235, 2.72,
    20, 39, 114,
    102, 667, 120,
    151, 190, 118
]

# popt, pcov = curve_fit(get_OLF, EE, OLF, p0=p0)
popt, pcov = curve_fit(get_OLF_log, EE_log, OLF_log, p0=p0)

plt.loglog(grid.EE, get_OLF(grid.EE, *popt))

plt.grid()

E1, A1, w1, E2, A2, w2, E3, A3, w3, E4, A4, w4 = popt
# Ei = 16.7, 20, 107.98, 151.55, 1828.5

OLF_fit = get_OSC(grid.EE, E1, A1, w1) + get_OSC_edge(grid.EE, E2, A2, w2) +\
    get_OSC_edge(grid.EE, E3, A3, w3) + get_OSC_edge(grid.EE, E4, A4, w4)

# plt.loglog(grid.EE, get_OSC(grid.EE, E1, A1, w1))
# plt.loglog(grid.EE, get_OSC_edge(grid.EE, E2, A2, w2))
# plt.loglog(grid.EE, get_OSC_edge(grid.EE, E3, A3, w3))
# plt.loglog(grid.EE, get_OSC_edge(grid.EE, E4, A4, w5))
# plt.loglog(grid.EE, get_OSC_edge(grid.EE, E5, A5, w4))

plt.loglog(grid.EE, OLF_fit)

# plt.ylim(1e-6, 1e+1)
plt.grid()

plt.show()


