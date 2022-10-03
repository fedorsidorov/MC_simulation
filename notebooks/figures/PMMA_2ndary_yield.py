import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const

const = importlib.reload(const)

# plt.rcParams.update({'font.size': 12})

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
            DATA_outer_2nd = DATA_outer[np.where(DATA_outer[:, -1] < 50)]

            n_2nd += len(DATA_outer_2nd)
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total

        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

    return E_final_list, d_list


# %%
folder_nf_0p02 = '/Volumes/Transcend/2ndaries/no_factor/0.02/'
energies_delta_nf_0p02 = get_2ndary_yield(folder_nf_0p02)


# %%
# plt.figure(dpi=600, figsize=[6.4, 4.8])
plt.figure(dpi=600, figsize=[6.4 / 1.9, 4.8 / 1.9])

# plt.plot(D_sim[:, 0], D_sim[:, 1], '.--', label='статья Дапора')
plt.plot(D_exp[:, 0], D_exp[:, 1], '.--', label='эксперимент')
plt.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], '.--', label='моделирование')

# plt.plot(D_exp[:, 0], D_exp[:, 1], 'o', label='эксперимент')
# plt.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], '*', label='моделирование')

plt.xlabel(r'$E$, эВ')
plt.ylabel(r'$\delta$')
# plt.text(-300, 2.16, 'a)', fontsize=12)

plt.xlim(0, 1500)
plt.ylim(0, 2.5)

plt.legend()
plt.grid()

# plt.savefig('figures/2ndary_yield.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('figures/2ndary_yield_1p9.jpg', dpi=600, bbox_inches='tight')
plt.savefig('figures/2ndary_yield_1p9.pdf', bbox_inches='tight')

plt.show()


# %%with plt.style.context(['science', 'grid', 'russian-font']):
# #     fig, ax = plt.subplots(dpi=600)
# #     ax.plot(D_sim[:, 0], D_sim[:, 1], 'b.--', label='статья Дапора')
# #     ax.plot(D_exp[:, 0], D_exp[:, 1], 'g.--', label='эксперимент')
# #     ax.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], 'r.--', label='моделирование')
# #
# #     # ax.legend(title=r'Число', fontsize=7)
# #     ax.legend(fontsize=7)
# #     ax.legend(fontsize=7)
# #     ax.set(xlabel=r'энергия электрона, эВ')
# #     ax.set(ylabel=r'выход вторичных электронов')
# #     ax.autoscale(tight=True)
# #     # plt.xlim(0.7, 1.2)
# #     plt.ylim(0, 2.5)
# #     # fig.savefig('figures/PMMA_2ndaries_new.jpg', dpi=600)
# #     plt.show()
#
