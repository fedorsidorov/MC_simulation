import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as cp
from mapping import mapping_harris as mapping
# import mapping_aktary as _outdated
from functions import array_functions as af
from functions import chain_functions as cf

cf = importlib.reload(cf)
af = importlib.reload(af)
cp = importlib.reload(cp)
mapping = importlib.reload(mapping)

# %%
source_dir = '/Volumes/ELEMENTS/Spyder/Chains/'
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

mw = np.load('data/mw_distributions/mw_harris.npy').astype(int)
mw_probs_n = np.load('data/mw_distributions/mw_probs_n_harris.npy')

# plt.figure(dpi=300)
# plt.semilogx(mw, mw_probs_n, 'ro')
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

    now_chain_len = int(np.random.choice(mw, p=mw_probs_n) / 100)
    chain_lens_list.append(now_chain_len)

    beg_ind = np.random.choice(len(now_chain_base) - now_chain_len)
    now_chain = now_chain_base[beg_ind:beg_ind + now_chain_len, :]
    chain_list.append(now_chain)

    n_monomers_now += now_chain_len
    progress_bar.update(now_chain_len)

chain_lens_array = np.array(chain_lens_list)

# %% save chains to files
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(chain_list):
    np.save('data/prepared_chains/Harris/chain_' + str(n) + '.npy', chain)
    progress_bar.update()

np.save('data/prepared_chains/Harris/chain_lens.npy', chain_lens_array)

# %%
print('Mn =', np.average(chain_lens_array) * 100)
print('Mw =', np.sum(chain_lens_array ** 2) / np.sum(chain_lens_array) * 100)

# %% check chain lengths distribution
# chain_lens_array = np.load('data/chains/' + folder_name + '/prepared_chains/prepared_chain_lens.npy')

simulated_mw_array = chain_lens_array * 100
bins = np.logspace(2, 7.1, 21)

plt.figure(dpi=300)
plt.hist(simulated_mw_array, bins, label='sample')
plt.gca().set_xscale('log')
# plt.plot(mw, molecular_weight_array * 0.6e+5, label='harris')
plt.xlabel('molecular weight')
plt.ylabel('density')
plt.xlim(1e+3, 1e+8)
plt.legend()
plt.grid()
plt.show()

# plt.savefig('Harris_sample_2020.png', dpi=300)
