import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as cp

# mapping parameters
# import mapping_harris as mapping
import mapping_aktary as mapping

from functions import array_functions as af
from functions import chain_functions as cf

cf = importlib.reload(cf)
af = importlib.reload(af)
cp = importlib.reload(cp)
mapping = importlib.reload(mapping)

# %%
source_dir = '/Volumes/ELEMENTS/Chains/'
chains = []

for folder in os.listdir(source_dir):
    if '.' in folder:
        continue

    print(folder)

    for _, fname in enumerate(os.listdir(os.path.join(source_dir, folder))):
        if 'DS_Store' in fname or '._' in fname:
            continue
        chains.append(np.load(os.path.join(source_dir, folder, fname)))

# %% create chain_list and check density
volume = np.prod(mapping.l_xyz) * cp.nm3_to_cm_3
n_monomers_required = int(cp.rho_PMMA * volume / cp.m_MMA)

mass_array = np.load('Resources/Harris/harris_x_before.npy')
molecular_weight_array = np.load('Resources/Harris/harris_y_before_fit.npy')
# plt.semilogx(mass_array, molecular_weight_array, 'ro')
# plt.show()

# %%
chain_list = []
chain_lens_list = []
n_monomers_now = 0
progress_bar = tqdm(total=n_monomers_required, position=0)

while True:
    if n_monomers_now > n_monomers_required:
        print('Needed density is achieved')
        break

    chain_base_ind = np.random.choice(len(chains))
    now_chain_base = chains[chain_base_ind]

    now_chain_len = cf.get_chain_len(mass_array, molecular_weight_array)
    chain_lens_list.append(now_chain_len)

    beg_ind = np.random.choice(len(now_chain_base) - now_chain_len)
    now_chain = now_chain_base[beg_ind:beg_ind + now_chain_len, :]
    chain_list.append(now_chain)

    n_monomers_now += now_chain_len
    progress_bar.update(now_chain_len)

chain_lens_array = np.array(chain_lens_list)

# %% save chains to files
folder_name = 'Aktary'
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(chain_list):
    np.save('data/chains/' + folder_name + '/prepared_chains/prepared_chain_' + str(n) + '.npy', chain)
    progress_bar.update()

np.save('data/chains/' + folder_name + '/prepared_chains/prepared_chain_lens.npy', chain_lens_array)

# %% check chain lengths distribution
chain_lens_array = np.load('data/chains/' + folder_name + '/prepared_chains/prepared_chain_lens.npy')

simulated_mw_array = chain_lens_array * 100
bins = np.logspace(2, 7.1, 21)

plt.figure(dpi=300)
plt.hist(simulated_mw_array, bins, label='sample')
plt.gca().set_xscale('log')
plt.plot(mass_array, molecular_weight_array * 0.6e+5, label='harris')
plt.xlabel('molecular weight')
plt.ylabel('density')
plt.xlim(1e+3, 1e+8)
plt.legend()
plt.grid()
plt.show()

# plt.savefig('Harris_sample_2020.png', dpi=300)
