import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
from tqdm import tqdm

const = importlib.reload(const)

# C_sim = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_MC/curves/ciappa2010.txt')
D_sim = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_MC/curves/Dapor_sim.txt')
D_exp = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/simple_MC/curves/Dapor_exp.txt')


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

            n_2nd += len(np.where(DATA[:, -1] < 50)[0])
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total
        
        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

        progress_bar.update()

    return E_final_list, d_list


# %%
energies_01_01, delta_01_01 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p1')
energies_01_02, delta_01_02 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p2')
energies_01_015, delta_01_015 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p15')

energies_corr, delta_corr = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p15_corr')
energies_factor, delta_factor = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p15_factor')
energies_uniform, delta_uniform = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p15_uniform')

energies_0, delta_0 = get_2ndary_yield('data/2ndaries/PMMA_outer_0')
energies_flat_lower, delta_flat_lower = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p15_flat_lower')
energies_flat_upper, delta_flat_upper = get_2ndary_yield('data/2ndaries/PMMA_outer_0p1_0p15_flat_upper')

energies_001_001, delta_001_001 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p01_0p01')
energies_001_005, delta_001_005 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p01_0p05')

energies_005_001, delta_005_001 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p05_0p01')
energies_005_005, delta_005_005 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p05_0p05')

energies_005_015, delta_005_015 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p05_0p15')
energies_015_015, delta_015_015 = get_2ndary_yield('data/2ndaries/PMMA_outer_0p15_0p15')

# %%
plt.figure(dpi=300)
# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o-', label='experiment')

plt.plot(energies_01_01, delta_01_01, '*-', label='C=0.1, g=0.1')
plt.plot(energies_01_015, delta_01_015, '*-', label='C=0.1, g=0.15')
plt.plot(energies_01_02, delta_01_02, '*-', label='C=0.1, g=0.2')

plt.plot(energies_corr, delta_corr, '*-', label='corr')
plt.plot(energies_factor, delta_factor, '*-', label='factor')
plt.plot(energies_uniform, delta_uniform, '*-', label='uniform')

plt.plot(energies_0, delta_0, '*-', label='0')
plt.plot(energies_flat_lower, delta_flat_lower, '*-', label='lower')
plt.plot(energies_flat_upper, delta_flat_upper, '*-', label='upper')

plt.plot(energies_001_001, delta_001_001, '*-', label='C=0.01, g=0.01')
plt.plot(energies_001_005, delta_001_005, '*-', label='C=0.01, g=0.05')

plt.plot(energies_005_001, delta_005_001, '*-', label='C=0.05, g=0.01')
plt.plot(energies_005_005, delta_005_005, '*-', label='C=0.05, g=0.05')

plt.plot(energies_005_015, energies_005_015, '*-', label='C=0.05, g=0.15')
plt.plot(energies_015_015, delta_015_015, '*-', label='C=0.15, g=0.15')

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
