import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const

const = importlib.reload(const)

# C_sim = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/PMMA_2ndary_check/ciappa2010.txt')
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
        # filenames = os.listdir(source)

        n_total = 0
        n_2nd = 0
        
        # for fname in filenames:
        for i in range(50):
            
            # if fname == '.DS_Store':
            #     continue
            
            # DATA_outer = np.load(os.path.join(source, fname))
            DATA_outer = np.load(os.path.join(source, 'e_DATA_' + str(i) + '.npy'))

            if n_primaries == 0:
                break

            DATA_outer = DATA_outer[np.where(DATA_outer[:, 5] < 0)]
            # DATA_outer_2nd = DATA_outer[np.where(
            #     np.logical_and(
            #         DATA_outer[:, -1] > 0,
            #         DATA_outer[:, -1] < 50))]

            n_2nd += len(DATA_outer)
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total
        
        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

    return E_final_list, d_list


# %%
now_folder = 'data/2ndaries/0.02'

energies, delta = get_2ndary_yield(now_folder)

plt.figure(dpi=300)
# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o-', label='experiment')

plt.plot(energies, delta, '*-', label='my simulation')

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

# plt.xlim(0, 1600)
# plt.ylim(0, 3)

plt.show()
# plt.savefig('2ndary_yield.jpg', dpi=300)
