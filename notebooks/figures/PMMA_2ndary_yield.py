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
        if E_str == '.DS_Store':
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

            if fname == '.DS_Store':
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
# folder_0 = 'data/2ndaries//0'

# folder_0p001 = 'data/2ndaries/0.001'
# folder_0p008 = 'data/2ndaries/0.008'
# folder_0p010 = 'data/2ndaries/0.01'
# folder_0p012 = 'data/2ndaries/0.012'
# folder_0p015 = 'data/2ndaries/0.015'
# folder_0p020 = 'data/2ndaries/0.02'
# folder_0p050 = 'data/2ndaries/0.05'

# folder_0p1 = 'data/2ndaries/0.1'
# folder_0p2 = 'data/2ndaries/0.2'
# folder_0p3 = 'data/2ndaries/0.3'

folder_nf_0p1 = 'data/2ndaries/no_factor/0.1/'
folder_nf_0p01 = 'data/2ndaries/no_factor/0.01/'
folder_nf_0p02 = 'data/2ndaries/no_factor/0.02/'
folder_nf_0p025 = 'data/2ndaries/no_factor/0.025/'
folder_nf_0p03 = 'data/2ndaries/no_factor/0.03/'
folder_nf_0p04 = 'data/2ndaries/no_factor/0.04/'
folder_nf_0p05 = 'data/2ndaries/no_factor/0.05/'
folder_nf_0p05_new = 'data/2ndaries/no_factor/0.05_new/'
folder_nf_0p075 = 'data/2ndaries/no_factor/0.075/'

# now_folder = 'data/2ndaries/log_rise_1e-5'

# energies_delta_0 = get_2ndary_yield(folder_0)

# energies_delta_0p001 = get_2ndary_yield(folder_0p001)
# energies_delta_0p008 = get_2ndary_yield(folder_0p008)
# energies_delta_0p010 = get_2ndary_yield(folder_0p010)
# energies_delta_0p012 = get_2ndary_yield(folder_0p012)
# energies_delta_0p015 = get_2ndary_yield(folder_0p015)
# energies_delta_0p020 = get_2ndary_yield(folder_0p020)
# energies_delta_0p050 = get_2ndary_yield(folder_0p050)

# energies_delta_0p1 = get_2ndary_yield(folder_0p1)
# energies_delta_0p2 = get_2ndary_yield(folder_0p2)
# energies_delta_0p3 = get_2ndary_yield(folder_0p3)

energies_delta_nf_0p1 = get_2ndary_yield(folder_nf_0p1)
energies_delta_nf_0p01 = get_2ndary_yield(folder_nf_0p01)
energies_delta_nf_0p02 = get_2ndary_yield(folder_nf_0p02)
energies_delta_nf_0p025 = get_2ndary_yield(folder_nf_0p025)
energies_delta_nf_0p03 = get_2ndary_yield(folder_nf_0p03)
energies_delta_nf_0p04 = get_2ndary_yield(folder_nf_0p04)
energies_delta_nf_0p05 = get_2ndary_yield(folder_nf_0p05)
energies_delta_nf_0p05_new = get_2ndary_yield(folder_nf_0p05_new)
energies_delta_nf_0p075 = get_2ndary_yield(folder_nf_0p075)

# %%
plt.figure(dpi=300)
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o--', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'go--', label='experiment')

# plt.plot(energies_delta_nf_0p01[0], energies_delta_nf_0p01[1], '*-', label='my simulation NF 0.01')
plt.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], 'r*-', label='my simulation NF 0.02')
# plt.plot(energies_delta_nf_0p025[0], energies_delta_nf_0p025[1], '*-', label='my simulation NF 0.025')
# plt.plot(energies_delta_nf_0p03[0], energies_delta_nf_0p03[1], '*-', label='my simulation NF 0.03')
# plt.plot(energies_delta_nf_0p04[0], energies_delta_nf_0p04[1], '*-', label='my simulation NF 0.04')
# plt.plot(energies_delta_nf_0p05_new[0], energies_delta_nf_0p05_new[1], '*-', label='my simulation NF 0.05')
# plt.plot(energies_delta_nf_0p06[0], energies_delta_nf_0p06[1], '*-', label='my simulation NF 0.06')
# plt.plot(energies_delta_nf_0p075[0], energies_delta_nf_0p075[1], '*-', label='my simulation NF 0.075')

# plt.plot(energies_delta_nf_0p1[0], energies_delta_nf_0p1[1], '*-', label='my simulation NF 0.1')

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

plt.xlim(0, 1600)
plt.ylim(0, 2.5)

plt.show()
# plt.savefig('2ndary_yield.jpg', dpi=300)

# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=300)
    # for p in [5, 7, 10, 15, 20, 30, 38, 50, 100]:
    #     ax.plot(x, model(x, p), label=p)

    ax.plot(D_sim[:, 0], D_sim[:, 1], 'o--', label='статья Дапора')
    ax.plot(D_exp[:, 0], D_exp[:, 1], 'go--', label='эксперимент')
    ax.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], 'r*-', label='моделировние')

    # ax.legend(title=r'Число', fontsize=7)
    ax.legend(fontsize=7)
    ax.set(xlabel=r'энергия электрона, эВ')
    ax.set(ylabel=r'выход вторичных электронов')
    ax.autoscale(tight=True)
    # plt.xlim(0.7, 1.2)
    plt.ylim(0, 2.5)
    # fig.savefig('fig16.jpg', dpi=300)
    plt.show()