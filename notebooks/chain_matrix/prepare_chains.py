import importlib
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import chain_functions as cf
from functions import array_functions as af
from tqdm import tqdm
import constants_mapping as const_m
import constants_physics as const_p

cf = importlib.reload(cf)
af = importlib.reload(af)
const_m = importlib.reload(const_m)
const_p = importlib.reload(const_p)

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
volume = np.prod(const_m.l_xyz) * const_m.nm3_to_cm_3
n_monomers_required = int(const_p.rho_PMMA * volume / const_p.m_MMA)

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

# %%
lens_inuque = np.unique(chain_lens_list)
lens_entries = np.zeros(len(lens_inuque))

for i, chain_len in enumerate(lens_inuque):
    lens_entries[i] = len(np.where(chain_lens_list == lens_inuque[i])[0])

# %% save chains to files
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(chain_list):
    np.save('data/Harris/prepared_chains/prepared_chain_' + str(n) + '.npy', chain)
    progress_bar.update(1)

np.save('data/Harris/prepared_chain_lens_array.npy', chain_lens_array)


# %% check chain lengths distribution
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
