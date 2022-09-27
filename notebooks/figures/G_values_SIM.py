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
TT = np.linspace(0, 200, 100)
GG = Gf.get_G(TT)

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)
    ax.plot(TT, GG, '-')

    ax.set(xlabel=r'T, °C')
    ax.set(ylabel=r'G value')

    plt.xlim(0, 200)
    plt.ylim(1, 3)

    # ax.autoscale(tight=True)
    plt.show()
    # fig.savefig('figures/G_value_T.jpg', dpi=600)


# %%
weights = [
    0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
    0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.135
]

GG_sim = np.zeros(len(weights))
GG_theor = np.zeros(len(weights))

n = 0
sample = '3'

for i, weight in enumerate(weights):

    scission_matrix = np.load('/Volumes/Transcend/G_calibration/' + sample + '/scission_matrix_' + str(weight) + '.npy')
    e_matrix_E_dep = np.load('data/e_matrix_E_dep.npy')
    chain_lens_initial = np.load('data/chain_lens.npy')
    chain_lens_final = np.load('/Volumes/Transcend/G_calibration/' + sample + '/harris_lens_final_' + str(weight) + '.npy')

    Mn = np.average(chain_lens_initial) * const.u_MMA
    Mf = np.average(chain_lens_final) * const.u_MMA

    total_E_loss = np.sum(e_matrix_E_dep)

    GG_sim[i] = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
    GG_theor[i] = np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100

TT_sim = Gf.get_T(GG_sim)
TT_theor = Gf.get_T(GG_theor)

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(TT_theor, weights, '.--', label='theory')
    ax.plot(TT_sim, weights, 'r.--', label='simulation')

    ax.legend(fontsize=7)
    ax.set(xlabel=r'T, °C')
    ax.set(ylabel=r'scission weight')

    plt.xlim(0, 200)
    plt.ylim(0.02, 0.12)

    plt.show()
    fig.savefig('figures/sci_weight_T.jpg', dpi=600)


