import importlib

import numpy as np
from scipy import integrate

import grid as g
from SimClasses import utilities as u, constants as c, arrays as a

a = importlib.reload(a)
c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)


# %%
def get_Gryzinski_DCS(Eb, E, hw):
    if E < hw or hw < Eb:
        return 0
    diff_cs = np.pi * c.e ** 4 / np.power(hw * c.eV, 3) * Eb / E * \
              np.power(E / (E + Eb), 3 / 2) * np.power((1 - hw / E), Eb / (Eb + hw)) * \
              (hw / Eb * (1 - Eb / E) + 4 / 3 * np.log(2.7 + np.sqrt((E - hw) / Eb)))
    return diff_cs * c.eV  # cm^2 / eV


def get_Gryzinski_CS(Eb, E):
    def get_Y(hw):
        return get_Gryzinski_DCS(Eb, E, hw)
    return integrate.quad(get_Y, Eb, (E + Eb) / 2)[0]


def get_Gryzinski_DIIMFP(Eb, E, hw, conc, n_el):
    return get_Gryzinski_DCS(Eb, E, hw) * conc * n_el


def get_Gryzinski_IIMFP(Eb, E, conc, n_el):
    return get_Gryzinski_CS(Eb, E) * conc * n_el
