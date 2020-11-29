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
TT = np.linspace(0, 170, 100)
GG = Gf.get_G(TT)

# plt.figure(dpi=300)
# plt.plot(TT, GG, 'o-')
# plt.grid()
# plt.show()

# %%
weights = 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325
GG_sim = np.zeros(len(weights))
GG_theor = np.zeros(len(weights))

sample = '2'

plt.figure(dpi=300)

markers = 'v-', '^-', '*-'

for n, sample in enumerate(['1', '2', '3']):

    for i, weight in enumerate(weights):

        scission_matrix = np.load('data/scission_mat_weight/' + sample + '/e_matrix_scissions_' + str(weight) + '.npy')
        e_matrix_E_dep = np.load('data/scission_mat_weight/' + sample + '/e_matrix_dE_' + str(weight) + '.npy')
        chain_lens_initial = np.load('/Volumes/ELEMENTS/chains_harris/prepared_chains_' + sample + '/chain_lens.npy')
        chain_lens_final = np.load('data/G_calibration/' + sample + '/harris_lens_final_' + str(weight) + '.npy')

        Mn = np.average(chain_lens_initial) * const.u_MMA
        Mf = np.average(chain_lens_final) * const.u_MMA

        total_E_loss = np.sum(e_matrix_E_dep)

        GG_sim[i] = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
        GG_theor[i] = np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100

    TT_sim = Gf.get_T(GG_sim)
    TT_theor = Gf.get_T(GG_theor)

    plt.plot(TT_sim, weights, markers[n], markersize=10, label='simulation ' + str(n))
    plt.plot(TT_theor, weights, markers[n], markersize=10, label='theory ' + str(n))
    plt.grid()

plt.xlim(0, 200)
# plt.ylim(0.254, 0.256)
plt.xlabel('T, °C')
plt.ylabel('w (remote scission probability)')
plt.legend()

plt.show()
# plt.savefig('G.png', dpi=300)
