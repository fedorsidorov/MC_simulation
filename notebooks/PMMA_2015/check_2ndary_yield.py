import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
import grid as grid
from tqdm import tqdm

const = importlib.reload(const)
grid = importlib.reload(grid)

D_sim = np.loadtxt('notebooks/PMMA_2015/curves/2ndaries_2015.txt')


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

            # DATA = DATA[np.where(DATA[:, 5] < 0)]

            if n_primaries == 0:
                break

            # n_2nd += len(np.where(DATA[:, -1] < 50)[0])
            n_2nd += len(np.where(DATA[:, -1] < 500)[0])
            n_total += n_primaries

        my_2ndary_yield = n_2nd / n_total
        
        E_final_list.append(E)
        d_list.append(my_2ndary_yield)

        progress_bar.update()

    return E_final_list, d_list


def get_2ndary_yield_1E(folder, n_files, n_primaries=100):

    n_2nd = 0
    n_total = 0

    for i in range(n_files):

        DATA = np.load(os.path.join(folder, 'e_DATA_' + str(i) + '.npy'))

        # n_2nd += len(np.where(DATA[:, 6] < 50)[0])
        n_2nd += len(DATA[:, 6])
        n_total += n_primaries

    my_2ndary_yield = n_2nd / n_total

    return my_2ndary_yield


# %%
energies, delta = get_2ndary_yield('data/2ndaries/2ndaries_outer_Wf_1_2015/')
# get_2ndary_yield_1E('data/2ndaries/PMMA_outer_2015/250', 100)

# %%
plt.figure(dpi=300)
# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')

plt.plot(energies, delta, '*-', label='my')

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

plt.xlim(0, 1600)
plt.ylim(0, 2.5)

plt.show()
# plt.savefig('2ndary_yield_models.jpg', dpi=300)

# %%
ans = np.load('data/2ndaries/2ndaries_outer_2011/200/e_DATA_2.npy')



