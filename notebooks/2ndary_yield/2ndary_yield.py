import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import constants as mc

mc = importlib.reload(mc)


#%%
def print_2ndaries(model, n_tracks):
    
    source_folder = os.path.join('data/2ndaries', model)
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
            
            DATA = np.load(os.path.join(source, fname))

            n_total += n_tracks
            n_2nd += DATA
        
        my_d = n_2nd/n_total
        
        E_final_list.append(E)
        d_list.append(my_d)

    plt.plot(E_final_list, d_list, '--', label='simulation')
    
    return E_final_list, d_list


#%%
# model = '0p1_0p15_chi_1_RH'
# model = '0p5_0p14_chi_1'
# model = '0p5_0p14_chi_1_RH'
# model = 'Mermin'

# model = '_outdated/0p1'
# model = '_outdated/0p1_0p1'
# model = '_outdated/0p2_0p1'
# model = '_outdated/0p3_0p2'
# model = '_outdated/0p5_0p14_chi_1'
# model = '_outdated/0p5_0p14_easy'

model_1 = '_outdated/0p07_0p1'  # close
model_2 = '_outdated/0p25'  # close

# model = '_outdated/0p25_0p05'
# model = '_outdated/1_0p5'
# model = '_outdated/1_0p14'
# model = '_outdated/1_0p14_extrap_easy'
# model = '_outdated/1_1'
# model = '_outdated/3_0p5'
# model = '_outdated/5_1'
# model = '_outdated/5_1'

plt.figure(dpi=300)

# C_sim = np.loadtxt('notebooks/2ndary_yield/ciappa2010.txt')
D_sim = np.loadtxt('notebooks/2ndary_yield/Dapor_sim.txt')
D_exp = np.loadtxt('notebooks/2ndary_yield/Dapor_exp.txt')

# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o-', label='experiment')

print_2ndaries(model_1, 100)
print_2ndaries(model_2, 100)

plt.xlabel('incident e energy, eV')
plt.ylabel('secondary electron yield')

plt.legend()
plt.grid()

# plt.xlim(0, 1600)
plt.ylim(0, 3)

plt.show()
