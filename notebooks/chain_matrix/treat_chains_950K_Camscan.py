import importlib
import os
from collections import deque

import numpy as np
from tqdm import tqdm

import constants as cp
from mapping import mapping_3p3um_80nm as mapping
from functions import array_functions as af
from functions import chain_functions as cf

cf = importlib.reload(cf)
af = importlib.reload(af)
cp = importlib.reload(cp)
mapping = importlib.reload(mapping)


# %%
def get_hist_position(element, bins):
    n_bin = (element - bins[0]) // (bins[1] - bins[0])
    return int(n_bin)


def get_shift(x_b, y_b, z_b):
    shift = [
        (mapping.x_bins_2nm[x_b] + mapping.x_bins_2nm[x_b + 1]) / 2,
        (mapping.y_bins_2nm[y_b] + mapping.y_bins_2nm[y_b + 1]) / 2,
        (mapping.z_bins_2nm[z_b] + mapping.z_bins_2nm[z_b + 1]) / 2
    ]
    return shift


def get_zero_shift(array):
    inds_xyz = np.array(np.where(array == np.min(array))).transpose()
    xyz_bin = inds_xyz[np.random.choice(len(inds_xyz))]
    return get_shift(*xyz_bin)


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
mw = np.load('data/mw_distributions/mw_950K.npy').astype(int)
mw_probs_n = np.load('data/mw_distributions/mw_probs_n_950K.npy')

volume = np.prod(mapping.l_xyz) * cp.nm3_to_cm_3
n_monomers_required = int(cp.rho_PMMA * volume / cp.m_MMA)

# plt.figure(dpi=300)
# plt.semilogx(mw, mw_probs_n, 'ro')
# plt.show()

# %%
chain_num = 0
chain_lens_deque = deque()
n_monomers_now = 0
hist_2nm = np.zeros(mapping.hist_2nm_shape)

shape = mapping.hist_2nm_shape

resist_matrix = deque(
    deque(
        deque(
            deque() for k in range(shape[2])
        ) for j in range(shape[1])
    ) for i in range(shape[0])
)

progress_bar = tqdm(total=n_monomers_required, position=0)

while True:

    if n_monomers_now > n_monomers_required:
        print('Needed density is achieved')
        break

    chain_base_ind = chain_num % len(chains)
    chain_num += 1

    now_chain_base = chains[chain_base_ind]

    now_chain_len = int(np.random.choice(mw, p=mw_probs_n) / 100)
    n_monomers_now += now_chain_len
    chain_lens_deque.append(now_chain_len)

    beg_ind = np.random.choice(len(now_chain_base) - now_chain_len)
    now_chain = now_chain_base[beg_ind:beg_ind + now_chain_len, :]

    now_chain -= now_chain[0, :]
    now_chain -= (np.max(now_chain, axis=0) + np.min(now_chain, axis=0)) / 2

    a, b, g = np.random.random(3) * 2 * np.pi
    now_chain = cf.rotate_chain(now_chain, a, b, g)

    now_shift = get_zero_shift(hist_2nm)
    now_chain_shifted = now_chain + now_shift

    af.snake_array(now_chain_shifted, 0, 1, 2, mapping.xyz_min, mapping.xyz_max)

    hist_2nm += np.histogramdd(now_chain_shifted, bins=mapping.bins_2nm)[0]
    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint32)

    for n_mon, mon_line in enumerate(now_chain_shifted):

        if len(now_chain_shifted) == 1:
            mon_type = 10
        elif n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain_shifted) - 1:
            mon_type = 2
        else:
            mon_type = 1

        x_bin = get_hist_position(element=mon_line[0], bins=mapping.x_bins_2nm)
        y_bin = get_hist_position(element=mon_line[1], bins=mapping.y_bins_2nm)
        z_bin = get_hist_position(element=mon_line[2], bins=mapping.z_bins_2nm)

        chain_table[n_mon] = x_bin, y_bin, z_bin, len(resist_matrix[x_bin][y_bin][z_bin]), mon_type
        resist_matrix[x_bin][y_bin][z_bin].append([chain_num, n_mon, mon_type])

    # np.save('/Volumes/ELEMENTS/chains_950K/chain_tables/exp_80nm_Camscan/chain_table_' +
    #         str(chain_num) + '.npy', chain_table)

    progress_bar.update(now_chain_len)


# %%
# for i in range(shape[0]):
#     print(i)
#     for j in range(shape[1]):
#         for k in range(shape[2]):
#             np.save('/Volumes/ELEMENTS/chains_950K/resist_matrix/exp_80nm_Camscan/resist_matrix_' +
#                     str(i) + '_' + str(j) + '_' + str(k) + '.npy', np.array(resist_matrix[i][j][k]))

# %% save chains to files
chain_lens_array = np.array(chain_lens_deque)
np.save('data/prepared_chains/harris/chain_lens.npy', chain_lens_array)

# %%
print('Mn =', np.average(chain_lens_array) * 100)
print('Mw =', np.sum(chain_lens_array ** 2) / np.sum(chain_lens_array) * 100)
