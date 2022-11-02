import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const

const = importlib.reload(const)

D_sim = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/PMMA_2ndary_check/Dapor_sim.txt')
D_exp = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/PMMA_2ndary_check/Dapor_exp.txt')


#%%
def get_2ndary_yield(source_folder, n_primaries=100):

    E_str_list = os.listdir(source_folder)
    E_list = []

    for E_str in E_str_list:
        if 'DS_Store' in E_str:
            continue

        E_list.append(int(E_str))

    E_final_list = []
    d_list = []

    for E in sorted(E_list):

        source = os.path.join(source_folder, str(E))
        filenames = os.listdir(source)

        n_total = 0
        n_2nd = 0

        for fname in filenames:

            if 'DS_Store' in fname:
                continue

            DATA_outer = np.load(os.path.join(source, fname))
            DATA_outer_2nd = DATA_outer[np.where(DATA_outer[:, 7] < 50)]

            n_2nd += len(DATA_outer_2nd)
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total

        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

    return E_final_list, d_list


# %%
# now_folder_0 = '/Volumes/Transcend/2ndaries/no_factor/0.02/'
# energies_delta_0 = get_2ndary_yield(now_folder_0)

now_folder = '/Volumes/Transcend/2ndaries/check/'
energies_delta = get_2ndary_yield(now_folder)

# %%
plt.figure(dpi=300)
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o--', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'go--', label='experiment')

# plt.plot(energies_delta_0[0], energies_delta_0[1], '*-', label='my simulation factor 0.02')
plt.plot(energies_delta[0], energies_delta[1], '*-', label='my simulation new')

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

plt.xlim(0, 1600)
plt.ylim(0, 2.5)

plt.show()
# plt.savefig('2ndary_yield.jpg', dpi=300)

# %%
# ans = np.load('/Volumes/Transcend/2ndaries/check/500/e_DATA_outer_0.npy')





