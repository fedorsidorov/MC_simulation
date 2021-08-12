import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import grid

grid = importlib.reload(grid)


# %% constants
d_PMMA = 100
arr_size = 1000


# %% energies
Wf_PMMA = 4.68
PMMA_E_cut = 3
PMMA_ee_E_bind = [0]

Si_E_pl = 16.7
# Si_E_cut = Si_E_pl
Si_E_cut = 30
Si_ee_E_bind = [0, 20.1, 102, 151.1, 1828.9]

E_cut = [PMMA_E_cut, Si_E_cut]
ee_E_bind = [PMMA_ee_E_bind, Si_ee_E_bind]

PMMA_elastic_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

PMMA_elastic_u[:E_10eV_ind] = PMMA_elastic_u[E_10eV_ind] * PMMA_elastic_mult

PMMA_elastic_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/PMMA/PMMA_diff_cs_cumulated_'
    + elastic_model + '_' + elastic_extrap + '+1.npy'
)

Si_elastic_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_'
    + elastic_model + '_u_' + elastic_extrap + 'nm.npy'
)

Si_elastic_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/elastic/final_arrays/Si/Si_diff_cs_cumulated_'
    + elastic_model + '_' + elastic_extrap + '+1.npy'
)

# e-e
PMMA_electron_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/u_nm.npy'
)

PMMA_electron_u_diff_cumulated = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Dapor_PMMA_Mermin/final_arrays/diff_u_cumulated.npy'
)

PMMA_electron_u_diff_cumulated[np.where(np.abs(PMMA_electron_u_diff_cumulated - 1) < 1e-10)] = 1

for i in range(arr_size):
    for j in range(arr_size - 1):

        if PMMA_electron_u_diff_cumulated[i, j] == 0 and PMMA_electron_u_diff_cumulated[i, j + 1] == 0:
            PMMA_electron_u_diff_cumulated[i, j] = -2

        if PMMA_electron_u_diff_cumulated[i, arr_size - j - 1] == 1 and \
                PMMA_electron_u_diff_cumulated[i, arr_size - j - 2] == 1:
            PMMA_electron_u_diff_cumulated[i, arr_size - j - 1] = -2

Si_electron_u = np.zeros((arr_size, 5))

for j in range(5):
    Si_electron_u[:, j] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u/u_' + str(j) + '_nm_precised.npy'
    )

Si_electron_u_diff_cumulated = np.zeros((5, arr_size, arr_size))

for n in range(5):
    Si_electron_u_diff_cumulated[n, :, :] = np.load(
        '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/u_diff_cumulated/u_diff_'
        + str(n) + '_cumulated_precised.npy'
    )

    Si_electron_u_diff_cumulated[np.where(np.abs(Si_electron_u_diff_cumulated - 1) < 1e-10)] = 1

    for i in range(arr_size):
        for j in range(arr_size - 1):

            if Si_electron_u_diff_cumulated[n, i, j] == 0 and Si_electron_u_diff_cumulated[n, i, j + 1] == 0:
                Si_electron_u_diff_cumulated[n, i, j] = -2

            if Si_electron_u_diff_cumulated[n, i, arr_size - j - 1] == 1 and \
                    Si_electron_u_diff_cumulated[n, i, arr_size - j - 2] == 1:
                Si_electron_u_diff_cumulated[n, i, arr_size - j - 1] = -2

    zero_inds = np.where(Si_electron_u_diff_cumulated[n, -1, :] == 0)[0]

    if len(zero_inds) > 0:

        zero_ind = zero_inds[0]

        if grid.EE[zero_ind] < Si_ee_E_bind[n]:
            Si_electron_u_diff_cumulated[n, :, zero_ind] = -2


Si_electron_u_diff_cumulated[0, :4, 5] = -2

Si_electron_u_diff_cumulated[1, :301, :297] = -2
Si_electron_u_diff_cumulated[1, 300, 296] = 0

Si_electron_u_diff_cumulated[2, :461, :457] = -2
Si_electron_u_diff_cumulated[2, 460, 456] = 0

Si_electron_u_diff_cumulated[3, :500, :496] = -2
Si_electron_u_diff_cumulated[3, 499, 495] = 0

Si_electron_u_diff_cumulated[4, :745, :742] = -2
Si_electron_u_diff_cumulated[4, 744, 741] = 0

# phonon, polaron
PMMA_phonon_u = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/arrays_PMMA/PMMA_ph_IMFP_nm.npy'
)

# 2015
C_polaron = 0.1  # nm^-1
gamma_polaron = 0.15  # eV^-1

PMMA_polaron_u = C_polaron * np.exp(-gamma_polaron * grid.EE)

# total u
PMMA_processes_u = np.vstack((PMMA_elastic_u, PMMA_electron_u, PMMA_phonon_u, PMMA_polaron_u)).transpose()
Si_processes_u = np.vstack((Si_elastic_u, Si_electron_u.transpose())).transpose()

# normed arrays
PMMA_processes_u_norm = np.zeros(np.shape(PMMA_processes_u))

for i in range(len(PMMA_processes_u)):
    if np.sum(PMMA_processes_u[i, :]) != 0:
        PMMA_processes_u_norm[i, :] = PMMA_processes_u[i, :] / np.sum(PMMA_processes_u[i, :])

PMMA_u_total = np.sum(PMMA_processes_u, axis=1)
PMMA_process_indexes = list(range(len(PMMA_processes_u[0, :])))


Si_processes_u_norm = np.zeros(np.shape(Si_processes_u))

for i in range(len(Si_processes_u)):
    if np.sum(Si_processes_u[i, :]) != 0:
        Si_processes_u_norm[i, :] = Si_processes_u[i, :] / np.sum(Si_processes_u[i, :])

Si_u_total = np.sum(Si_processes_u, axis=1)
Si_process_indexes = list(range(len(Si_processes_u[0, :])))

# structure process lists
processes_u = [PMMA_processes_u, Si_processes_u]
u_total = [PMMA_u_total, Si_u_total]
u_norm = [PMMA_processes_u_norm, Si_processes_u_norm]

el_u_diff_cumulated = [PMMA_elastic_u_diff_cumulated, Si_elastic_u_diff_cumulated]

process_indexes = [PMMA_process_indexes, Si_process_indexes]


# %%
layer_ind = 1

l_1 = 1 / u_total[layer_ind]
l_2 = 1 / u_total[1 - layer_ind]

E_ind = 900
d = 1

W1 = 1 / l_1[E_ind]
W2 = 1 / l_2[E_ind]

u1 = np.random.random()
free_path = - 1 / W1 * np.log(1 - u1)

if u1 < (1 - np.exp(- W1 * d)):
    free_path_corr = 1 / W1 * (-np.log(1 - u1))
    print('here')
else:
    free_path_corr = d + 1 / W2 * (-np.log(1 - u1) - W1 * d)

print(free_path, free_path_corr)

# %%
E_ind = 900
a1 = 10

W1 = 1 / l_1[E_ind]
W2 = 1 / l_2[E_ind]

s = np.linspace(0, 50, 100)

f = np.zeros(len(s))
F = np.zeros(len(s))

f[np.where(s < a1)[0]] = np.exp(-W1 * s[np.where(s < a1)[0]])
f[np.where(s >= a1)[0]] = np.exp(-W1 * a1) * np.exp(-W2 * (s[np.where(s >= a1)[0]] - a1))
f_1 = np.exp(-W1 * s)

F[np.where(s < a1)[0]] = 1 - np.exp(-W1 * s[np.where(s < a1)[0]])
F[np.where(s >= a1)[0]] = 1 - np.exp(-(W1 - W2) * a1 - W2 * s[np.where(s >= a1)[0]])
F_1 = 1 - np.exp(-W1 * s)

plt.figure(dpi=300)

plt.plot(s, f, 'o', label='f')
plt.plot(s, f_1, '-', label='f_1')
plt.plot(s, F, 'o', label='F')
plt.plot(s, F_1, '-', label='F_1')

plt.xlim(0, 50)
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

