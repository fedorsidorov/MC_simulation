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

plt.figure(dpi=300)
plt.plot(TT, GG, 'o-')
plt.grid()
plt.show()

# %%
# weight = '0.15'  # 1.4
# weight = '0.175'  # 1.6
# weight = '0.2'  # 1.8
# weight = '0.225'  # 2.0
# weight = '0.25'  # 2.18
# weight = '0.275'  # 2.37
# weight = '0.3'  # 2.55
weight = '0.325'  # 2.73

scission_matrix = np.load('data/scission_mat_weight/e_matrix_scissions_' + weight + '.npy')
e_matrix_E_dep = np.load('data/scission_mat_weight/e_matrix_dE_' + weight + '.npy')
chain_lens_initial = np.load('/Volumes/ELEMENTS/chains_harris/prepared_chains_1/chain_lens.npy')
chain_lens_final = np.load('data/G_calibration/harris_lens_final_' + weight + '.npy')

Mn = np.average(chain_lens_initial) * const.u_MMA
Mf = np.average(chain_lens_final) * const.u_MMA

total_E_loss = np.sum(e_matrix_E_dep)

G_scission = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
print('simulated G(S) =', G_scission)
print('direct G(S) =', np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100)

# %%
weights = 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325
GG_sim = np.zeros(len(weights))
GG_theor = np.zeros(len(weights))

sample = '2'

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

plt.figure(dpi=300)
plt.plot(TT_sim, weights, 'o-')
plt.plot(TT_theor, weights, 'o-')
plt.grid()

# plt.xlim(150, 170)
# plt.ylim(0.254, 0.256)

plt.show()

# %%
TT_exact = np.linspace(TT_sim[0], TT_sim[-1], 100)
weights_exact = mcf.lin_lin_interp(TT_sim, weights)(TT_exact)

ans = np.zeros((len(weights_exact), 2))

ans[:, 0] = TT_exact
ans[:, 1] = weights_exact

plt.figure(dpi=300)
plt.plot(TT_exact, weights_exact, 'o-')
plt.plot(TT_sim, weights, 'o-')
plt.grid()
plt.show()
