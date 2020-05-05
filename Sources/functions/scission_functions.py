import importlib
import numpy as np
import matplotlib.pyplot as plt

import constants as const
import grid

const = importlib.reload(const)
grid = importlib.reload(grid)


# %%
kJmol2eV = 1e+3 / (const.Na * const.eV_SI)
# kJmol_2_eV = 0.0103

MMA_bonds = {
    "Oval": (13.62, 8),
    "C'-O'": (815 * kJmol2eV, 4),
    "C'-O": (420 * kJmol2eV, 2),
    "C3-H": (418 * kJmol2eV, 12),
    "C2-H": (406 * kJmol2eV, 4),
    "C-C'": (373 * kJmol2eV, 2),  # 383-10
    "O-C3": (364 * kJmol2eV, 2),
    "C-C3": (356 * kJmol2eV, 2),
    "C-C2": (354 * kJmol2eV, 4)
}

n_bonds = len(MMA_bonds)
bond_inds = list(range(n_bonds))
bond_names = list(MMA_bonds.keys())
BDE_array = np.array(list(MMA_bonds.values()))
bonds_BDE = BDE_array[:, 0]
bonds_occ = BDE_array[:, 1]
Ebond_Nelectrons_array = np.array(list(MMA_bonds.values()))


# %%
def get_scission_probs(degpath_dict, E_array=grid.EE):
    Ebond_Nelectrons_scission_list = []

    for value in degpath_dict.keys():
        Ebond_Nelectrons_scission_list.append([MMA_bonds[value][0], degpath_dict[value]])

    Ebond_Nelectrons_scission_array = np.array(Ebond_Nelectrons_scission_list)

    probs = np.zeros(len(E_array))

    for i, E in enumerate(E_array):
        num = 0

        for Eb_Nel in Ebond_Nelectrons_scission_array:
            if E >= Eb_Nel[0]:
                num += Eb_Nel[1]

        if num == 0:
            continue

        den = 0

        for Eb_Nel in Ebond_Nelectrons_array:
            if E >= Eb_Nel[0]:
                den += Eb_Nel[1]

        probs[i] = num / den

    return probs


# %%
# EE = np.linspace(0, 10, 1000)
#
# stairway_RT = get_scission_probs({"C-C2": 4}, EE)
# stairway_Hi = get_scission_probs({"C-C2": 4, "C-C'": 2}, EE)
#
# plt.figure(dpi=300)
# plt.plot(EE, stairway_RT)
# plt.plot(EE, stairway_Hi)
# plt.grid()
# plt.show()


# %%
# def get_Gs_charlesby(T):
#     inv_T = 1000 / (T + 273)
#
#     k = -0.448036
#     b = 1.98906
#     #    b = 2.14
#
#     return np.exp(k * inv_T + b)

