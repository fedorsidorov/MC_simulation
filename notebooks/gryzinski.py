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
## PMMA           C,   O
# MMA_core_Eb = [296, 538]
# MMA_core_occ = [2, 2]
#
# N_val_MMA = 40
#
# N_H_MMA = 8
# N_C_MMA = 5
# N_O_MMA = 2
#
# n_MMA = mc.rho_PMMA * mc.Na / mc.u_MMA
#
# Si            1s,  2s,  2p
# Si_core_Eb = [1844, 154, 104]
# Si_core_occ = [2, 2, 6]
#
# Si             1s,  2s,  2p,    3s,      3p
# Si_total_Eb = [1844, 154, 104, 13.46, 8.15]
# Si_total_occ = [2, 2, 6, 2, 2]

# Si_MuElec_Eb = [16.65, 6.52, 13.63, 107.98, 151.55, 1828.5]
# Si_MuElec_occ = [4, 2, 2, 6, 2, 2]

#  energyConstant.push_back(16.65*eV);
#  energyConstant.push_back(6.52*eV);
#  energyConstant.push_back(13.63*eV);
#  energyConstant.push_back(107.98*eV);
#  energyConstant.push_back(151.55*eV);
#  energyConstant.push_back(1828.5*eV);

# N_val_Si = 4
#
# n_Si = mc.rho_Si * mc.Na / mc.u_Si


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
