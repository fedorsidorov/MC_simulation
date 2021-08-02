import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
from tqdm import tqdm

const = importlib.reload(const)

# C_sim = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/PMMA/curves/ciappa2010.txt')
D_sim = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/PMMA/curves/Dapor_sim.txt')
D_exp = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_PMMA_MC/PMMA/curves/Dapor_exp.txt')


#%%
def get_2ndary_yield(folder, n_primaries=100):
    
    E_str_list = os.listdir(folder)
    E_list = []

    for E_str in E_str_list:
        
        if E_str == '.DS_Store':
            continue
        
        E_list.append(int(E_str))
    
    E_final_list = []
    d_list = []

    progress_bar = tqdm(total=len(E_list), position=0)

    for E in sorted(E_list):
    
        source = os.path.join(folder, str(E))
        filenames = os.listdir(source)

        n_total = 0
        n_2nd = 0
        
        for fname in filenames:
            
            if fname == '.DS_Store':
                continue
            
            DATA = np.load(os.path.join(source, fname))

            DATA = DATA[np.where(DATA[:, 5] < 0)]

            if n_primaries == 0:
                break

            n_2nd += len(np.where(DATA[:, -1] < 45)[0])
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total
        
        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

        progress_bar.update()

    return E_final_list, d_list


# %%
energies_2011, delta_2011 = get_2ndary_yield('data/2ndaries/PMMA_outer_2011')

# %%
plt.figure(dpi=300)
# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o-', label='experiment')

plt.plot(energies_2011, delta_2011, '*-', label='2011')

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

plt.xlim(0, 1600)
plt.ylim(0, 2.5)

plt.show()
# plt.savefig('2ndary_yield_models.jpg', dpi=300)

# %%
ans = np.load('data/si_pmma_si/easy/100/e_DATA_0.npy')
