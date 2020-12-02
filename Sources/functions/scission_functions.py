import importlib

import numpy as np
import matplotlib.pyplot as plt
import constants as const
import grid
import indexes as ind

const = importlib.reload(const)
grid = importlib.reload(grid)
ind = importlib.reload(ind)

# %%
kJmol2eV = 1e+3 / (const.Na * const.eV_SI)
kJmol_2_eV = 1e+3 / (const.Na * const.eV)

MMA_bonds = {
    'Oval': (13.62, 8),  # 13.62
    'Cp-Op': (815 * kJmol2eV, 4),  # 8.45
    'Cp-O': (420 * kJmol2eV, 2),  # 4.35
    'C3-H': (418 * kJmol2eV, 12),  # 4.33
    'C2-H': (406 * kJmol2eV, 4),  # 4.21
    'CCp-Cp': (383 * kJmol2eV, 2),  # 3.97
    'O-C3': (364 * kJmol2eV, 2),  # 3.77
    'CCp-C3': (332 * kJmol2eV, 2),  # 3.44
    'CCp-C2': (329 * kJmol2eV, 4)  # 3.41
}

n_bonds = len(MMA_bonds)
bond_inds = list(range(n_bonds))
bond_names = list(MMA_bonds.keys())
BDE_array = np.array(list(MMA_bonds.values()))
bonds_BDE = BDE_array[:, 0]
bonds_occ = BDE_array[:, 1]
Ebond_Nelectrons_array = np.array(list(MMA_bonds.values()))

# %%
degpaths_all = {'Oval': 8, 'Cp-Op': 4, 'Cp-O': 2, 'C3-H': 12, 'C2-H': 4,
                'CCp-Cp': 2, 'O-C3': 2, 'CCp-C3': 2, 'CCp-C2': 4}
degpaths_all_WO_Oval = {'Cp-Op': 4, 'Cp-O': 2, 'C3-H': 12, 'C2-H': 4,
                        'CCp-Cp': 2, 'O-C3': 2, 'CCp-C3': 2, 'CCp-C2': 4}
degpaths_CC = {'CCp-C2': 4}
degpaths_CC_ester = {'CCp-C2': 4, 'CCp-Cp': 2}


# %% _outdated
# def get_scission_probs(degpath_dict, E_array=grid.EE):
#     Ebond_Nelectrons_scission_list = []
#
#     for value in degpath_dict.keys():
#         Ebond_Nelectrons_scission_list.append([MMA_bonds[value][0], degpath_dict[value]])
#
#     Ebond_Nelectrons_scission_array = np.array(Ebond_Nelectrons_scission_list)
#     probs = np.zeros(len(E_array))
#
#     for i, E in enumerate(E_array):
#
#         num = 0
#         for Eb_Nel in Ebond_Nelectrons_scission_array:
#             if E >= Eb_Nel[0]:
#                 num += Eb_Nel[1]
#
#         if num == 0:
#             continue
#
#         den = 0
#         for Eb_Nel in Ebond_Nelectrons_array:
#             if E >= Eb_Nel[0]:
#                 den += Eb_Nel[1]
#
#         probs[i] = num / den
#
#     return probs


# def get_scissions(DATA, degpath_dict, weight=1):
#     ee = DATA[:, ind.DATA_E_dep_ind] + DATA[:, ind.DATA_E2nd_ind] + DATA[:, ind.DATA_E_ind]  # E before collision
#     scission_probs = get_scission_probs(degpath_dict, E_array=ee) * weight
#     return np.array(np.random.random(len(DATA)) < scission_probs).astype(int)


# %%
def get_scissions(e_DATA, weight):
    ee = e_DATA[:, ind.e_DATA_E_dep_ind] + e_DATA[:, ind.e_DATA_E2nd_ind] + e_DATA[:, ind.e_DATA_E_ind]  # E before collision
    scission_probs = np.ones(len(ee)) * weight
    return np.array(np.random.random(len(e_DATA)) < scission_probs).astype(int)
