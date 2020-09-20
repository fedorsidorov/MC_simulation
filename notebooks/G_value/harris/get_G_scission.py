import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import constants as const
from mapping import mapping_harris as mapping

mapping = importlib.reload(mapping)
const = importlib.reload(const)


# %%
weight = '0.2'

e_matrix_E_dep = np.load('data/scission_mat_weight/e_matrix_dE_' + weight + '.npy')
chain_lens_initial = np.load('/Volumes/ELEMENTS/chains_harris/prepared_chains_1/chain_lens.npy')
chain_lens_final = np.load('data/G_calibration/harris_lens_final_' + weight + '.npy')

Mn = np.average(chain_lens_initial) * const.u_MMA
Mf = np.average(chain_lens_final) * const.u_MMA

total_E_loss = np.sum(e_matrix_E_dep)

G_scission = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
# G_exp[i] = G_scission
print('experimental G(S) =', G_scission)

# %% direct calculation
scission_matrix = np.load('data/scission_mat_weight/e_matrix_scissions_' + weight + '.npy')
print('direct G(S) =', np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100)
# ratio = G_scission / (np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100)
# ratios[i] = ratio
# print('ratio =', ratio)


# %%
# weights_int = list(range(150, 455, 5))
# weights = np.array(range(150, 455, 5)) / 1000
# G_exp_arr = np.zeros(len(weights))
#
# for i, w in enumerate(weights_int):
#
#     # weight = w / 1000
#
#     e_matrix_E_dep = np.load('data/choi_weight/e_matrix_dE_0.' + str(weights_int[i]) + '.npy')
#     scission_matrix = np.load('data/choi_weight/e_matrix_val_ion_sci_0.' + str(weights_int[i]) + '.npy')
#
#     G_theor = np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100
#     G_exp_arr[i] = G_theor * (-0.397025766464037 * weights[i] + 0.97749357187016882)
#
# plt.plot(weights, G_exp_arr, 'o')
# plt.show()
