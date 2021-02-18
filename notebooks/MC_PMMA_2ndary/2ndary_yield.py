import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const

const = importlib.reload(const)

# C_sim = np.loadtxt('notebooks/MC_PMMA_2ndary/ciappa2010.txt')
D_sim = np.loadtxt('notebooks/MC_PMMA_2ndary/Dapor_sim.txt')
D_exp = np.loadtxt('notebooks/MC_PMMA_2ndary/Dapor_exp.txt')


#%%
def get_2ndary_yield(model, n_primaries=100):
    
    source_folder = os.path.join('data/2ndaries', model)
    E_str_list = os.listdir(source_folder)
    # E_str_list = '100', '200'

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
            
            DATA = np.load(os.path.join(source, fname))

            if n_primaries == 0:
                break

            n_2nd += len(np.where(DATA[:, -1] < 50)[0])
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total
        
        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

    return E_final_list, d_list


# %%
# now_model = '0p1_0p15_0eV_4p68'
# now_model = '0p1_0p15_0eV_4p05'
now_model = '0p1_0p15_0eV_4p05_scale'
# now_model = '0p07_0p1_0eV'

energies, delta = get_2ndary_yield(now_model)

plt.figure(dpi=300)
# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o-', label='experiment')

plt.plot(energies, delta, '*-', label=now_model)

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

# plt.xlim(0, 1600)
plt.ylim(0, 3)

plt.show()

# %%
ans = np.load('data/2ndaries/1p5_0p14_0eV/100/e_DATA_outer_92.npy')
