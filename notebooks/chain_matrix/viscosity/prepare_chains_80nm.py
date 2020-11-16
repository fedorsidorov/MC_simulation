import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import constants as cp
from mapping import mapping_viscosity_80nm as mm
from functions import array_functions as af
from functions import chain_functions as cf

cf = importlib.reload(cf)
af = importlib.reload(af)
cp = importlib.reload(cp)
mm = importlib.reload(mm)

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
n_monomers_required = int(cp.rho_PMMA * mm.volume_cm3 / cp.M_mon)

mw = np.load('data/mw_distributions/mw_950K.npy').astype(int)
mw_probs_n = np.load('data/mw_distributions/mw_probs_n_950K.npy')

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
bin_size = '10nm'

progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(chain_list):
    np.save('/Volumes/ELEMENTS/chains_viscosity_80nm/' + bin_size + '/prepared_chains_1/chain_' + str(n) + '.npy', chain)
    progress_bar.update()

np.save('/Volumes/ELEMENTS/chains_viscosity_80nm/' + bin_size + '/prepared_chains_1/chain_lens.npy', chain_lens_array)

# %%
print('Mn =', np.average(chain_lens_array) * 100)
print('Mw =', np.sum(chain_lens_array ** 2) / np.sum(chain_lens_array) * 100)
