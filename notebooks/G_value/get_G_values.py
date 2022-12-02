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

plt.xlabel('T, °C')
plt.ylabel('G value')

plt.grid()
plt.show()

# %%
# weights = [
#     0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020,
#     0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030
# ]

weights = [
    0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
    0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.135
]

# weights = [
#     0.030, 0.080, 0.135
# ]

GG_sim = np.zeros(len(weights))
GG_theor = np.zeros(len(weights))

sample = '5'

# plt.figure(dpi=300)

# fontsize = 14

_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
# fig.set_size_inches(4, 3)
# fig.set_size_inches(6, 4.5)

markers = 'v-', '^-', '*-'

for n, sample in enumerate([sample]):

    for i, weight in enumerate(weights):

        # scission_matrix = np.load('/Volumes/Transcend/G_calibration/' + sample +
        #                           '/scission_matrix_' + str(weight) + '.npy')

        scission_matrix = np.load('/Volumes/TOSHIBA EXT/G_calibration/' + sample +
                                  '/scission_matrix_' + str(weight) + '.npy')

        e_matrix_E_dep = np.load('data/e_matrix_E_dep.npy')

        chain_lens_initial = np.load('data/chain_lens.npy')

        chain_lens_final = np.load('/Volumes/TOSHIBA EXT/G_calibration/' + sample +
                                   '/harris_lens_final_' + str(weight) + '.npy')

        Mn = np.average(chain_lens_initial) * const.u_MMA
        Mf = np.average(chain_lens_final) * const.u_MMA

        total_E_loss = np.sum(e_matrix_E_dep)

        GG_sim[i] = (Mn / Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100
        GG_theor[i] = np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100

    TT_sim = Gf.get_T(GG_sim)
    TT_theor = Gf.get_T(GG_theor)

    # plt.plot(TT_sim, weights, markers[n], markersize=10, label='simulation ' + str(n))
    # plt.plot(TT_theor, weights, markers[n], markersize=10, label='theory ' + str(n))

    # plt.plot(TT_sim, weights, markers[n], markersize=10, label='simulation')
    # plt.plot(TT_theor, weights, markers[n], markersize=10, label='theory')

    plt.plot(TT_sim, weights, 'r-o', linewidth=3, label='моделирование G$_S$ на основе M$_n$')
    plt.plot(TT_theor, weights, '--o', color='C0', linewidth=3, label='моделирование G$_S$ на основе E$_{dep}$')

# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(fontsize)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(fontsize)

plt.xlim(0, 200)
# plt.ylim(0.02, 0.12)
# plt.xlabel('T, °C', fontsize=fontsize)
plt.title('sample ' + sample)
plt.xlabel('T, °C')
plt.ylabel('вероятность разрыва (P$_S$)')
plt.legend()

plt.grid()

plt.savefig('notebooks/G_value/samples/G_value_' + sample + '.jpg', bbox_inches='tight')
np.save('notebooks/G_value/samples/TT_sim_' + sample + '.npy', TT_sim)
np.save('notebooks/G_value/samples/TT_theor_' + sample + '.npy', TT_theor)

# plt.show()

# %% Treat G-values
weights = np.load('notebooks/G_value/samples/weights.npy')

TT_1 = np.load('notebooks/G_value/samples/TT_sim_1.npy')
TT_2 = np.load('notebooks/G_value/samples/TT_sim_2.npy')
TT_3 = np.load('notebooks/G_value/samples/TT_sim_3.npy')
TT_4 = np.load('notebooks/G_value/samples/TT_sim_4.npy')
TT_5 = np.load('notebooks/G_value/samples/TT_sim_5.npy')

TT = (TT_1 + TT_2 + TT_3 + TT_4 + TT_5) / 5

# plt.figure(dpi=300)
# plt.plot(TT, weights)
# plt.show()

mcf.lin_lin_interp(TT, weights)(140)  # 0.08581784
mcf.lin_lin_interp(TT, weights)(145)  # 0.08725558
mcf.lin_lin_interp(TT, weights)(150)  # 0.08869333
mcf.lin_lin_interp(TT, weights)(155)  # 0.09013067
mcf.lin_lin_interp(TT, weights)(160)  # 0.09156405






