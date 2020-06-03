import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import constants as const
import mapping_harris as mapping

const = importlib.reload(const)
mapping = importlib.reload(mapping)


# %%
def linear_func(xx, k, b):
    return k * xx + b


weights = ['0.150', '0.175', '0.200', '0.225', '0.250', '0.275', '0.300', '0.320',
           '0.350', '0.375', '0.400', '0.425', '0.450']

weights_num = [float(w) for w in weights]

weights = ['all']

G_exp = np.zeros(len(weights))
ratios = np.zeros(len(weights))

for i, weight in enumerate(weights):
    e_matrix_E_dep = np.load('data/choi_weight/e_matrix_dE_' + weight + '.npy')
    chain_lens_initial = np.load('data/choi_weight/harris_lens_initial.npy')
    chain_lens_final = np.load('data/choi_weight/harris_lens_final_' + weight + '.npy')

    Mn = np.average(chain_lens_initial) * const.u_MMA
    Mf = np.average(chain_lens_final) * const.u_MMA

    total_E_loss = np.sum(e_matrix_E_dep)
    G_scission = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
    G_exp[i] = G_scission
    print('experimental G(S) =', G_scission)

    # direct calculation
    scission_matrix = np.load('data/choi_weight/e_matrix_val_ion_sci_' + weight + '.npy')
    print('direct G(S) =', np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100)
    # ratio = G_scission / (np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100)
    # ratios[i] = ratio
    # print('ratio =', ratio)

# %%
popt, pcov = curve_fit(linear_func, weights_num, ratios)

plt.plot(weights_num, ratios, 'o')
plt.plot(weights_num, linear_func(np.array(weights_num), *popt))
# plt.plot(weights_num, linear_func(np.array(weights_num), -0.397025766464037, 0.97749357187016882))

plt.show()

# %%
weights_int = list(range(150, 455, 5))
weights = np.array(range(150, 455, 5)) / 1000
G_exp_arr = np.zeros(len(weights))

for i, w in enumerate(weights_int):

    # weight = w / 1000

    e_matrix_E_dep = np.load('data/choi_weight/e_matrix_dE_0.' + str(weights_int[i]) + '.npy')
    scission_matrix = np.load('data/choi_weight/e_matrix_val_ion_sci_0.' + str(weights_int[i]) + '.npy')

    G_theor = np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100
    G_exp_arr[i] = G_theor * (-0.397025766464037 * weights[i] + 0.97749357187016882)

plt.plot(weights, G_exp_arr, 'o')
plt.show()
