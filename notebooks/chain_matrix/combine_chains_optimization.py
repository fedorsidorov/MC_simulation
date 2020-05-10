import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
# import mapping_harris as mapping
from collections import deque
import mapping_aktary as mapping
import constants as cp

cf = importlib.reload(cf)
af = importlib.reload(af)
pf = importlib.reload(pf)
mapping = importlib.reload(mapping)
cp = importlib.reload(cp)


# %%
def get_shift(x_bin, y_bin, z_bin):
    shift = [
        (mapping.x_bins_2nm[x_bin] + mapping.x_bins_2nm[x_bin + 1]) / 2,
        (mapping.y_bins_2nm[y_bin] + mapping.y_bins_2nm[y_bin + 1]) / 2,
        (mapping.z_bins_2nm[z_bin] + mapping.z_bins_2nm[z_bin + 1]) / 2
    ]
    return shift


def get_zero_shift(array):
    inds_xyz = np.array(np.where(array == 0)).transpose()
    xyz_bin = inds_xyz[np.random.choice(len(inds_xyz))]
    return get_shift(*xyz_bin)


# %%
source_dir = '/Volumes/ELEMENTS/Chains/'
chains = deque()

for folder in os.listdir(source_dir):
    if '.' in folder:
        continue

    print(folder)

    for _, fname in enumerate(os.listdir(os.path.join(source_dir, folder))):
        if 'DS_Store' in fname or '._' in fname:
            continue
        chains.append(np.load(os.path.join(source_dir, folder, fname)))

# %%
best_chains = deque()
best_part_empty = 1
best_max = np.inf
best_hist_2nm = np.zeros(mapping.hist_2nm_shape)
hist_2nm = np.zeros(mapping.hist_2nm_shape)

n_iterations = 2200

progress_bar = tqdm(total=n_iterations, position=0)

for _ in range(n_iterations):

    chain_deque = []
    n_monomers_now = 0

    for _ in range(754):

        chain_base_ind = np.random.choice(len(chains))
        now_chain_base = chains[chain_base_ind]
        now_chain_len = 9500

        beg_ind = np.random.choice(len(now_chain_base) - now_chain_len)
        now_chain = now_chain_base[beg_ind:beg_ind + now_chain_len, :]
        chain_deque.append(now_chain)

    chain_lens_array = np.array(754)

    final_chain_deque = deque()
    n_monomers_now = 0

    hist_2nm = np.zeros(mapping.hist_2nm_shape)

    for n, now_chain in enumerate(chain_deque):

        now_chain -= now_chain[0, :]
        now_chain_center_xyz = (np.max(now_chain, axis=0) + np.min(now_chain, axis=0)) / 2
        now_chain = now_chain - now_chain_center_xyz

        a, b, g = np.random.random(3) * 2 * np.pi
        now_chain = cf.rotate_chain(now_chain, a, b, g)

        now_chain_l_xyz = np.max(now_chain, axis=0) - np.min(now_chain, axis=0)
        now_shift = get_zero_shift(hist_2nm)
        now_chain_shifted = now_chain + now_shift

        af.snake_array(now_chain_shifted, 0, 1, 2, mapping.xyz_min, mapping.xyz_max)
        final_chain_deque.append(now_chain_shifted)

        hist_2nm += np.histogramdd(now_chain_shifted, bins=mapping.bins_2nm)[0]

    # n_empty = np.prod(mapping.hist_2nm_shape) - np.count_nonzero(hist_2nm)
    # part_empty = n_empty / np.prod(mapping.hist_2nm_shape)

    # if part_empty < best_part_empty:
    #     best_chains = final_chain_deque
    #     best_part_empty = part_empty

    now_max = np.max(hist_2nm)

    if now_max < best_max:
        best_chains = final_chain_deque
        best_max = now_max
        best_hist_2nm = hist_2nm

    progress_bar.update()

    # print('\nbest part of empty bins =', best_part_empty)
    print('\nbest max of hist =', best_max)

# %%
plt.imshow(np.average(best_hist_2nm, axis=1))
plt.show()

# %%
n_empty = np.prod(mapping.hist_2nm_shape) - np.count_nonzero(best_hist_2nm)
part_empty = n_empty / np.prod(mapping.hist_2nm_shape)

print(np.sqrt(np.var(best_hist_2nm)))
print(part_empty)

# %%
progress_bar = tqdm(total=len(best_chains), position=0)

for n, chain in enumerate(best_chains):
    np.save('data/chains/Aktary/best_sh_sn_chains/sh_sn_chain_' + str(n) + '.npy', chain)
    progress_bar.update()

# %%
np.save('data/chains/Aktary/best_sh_sn_chains/best_hist_2nm.npy', best_hist_2nm)

# %% save chains to files
# data/chains/Aktary/shifted_snaked_chains
dest_folder = 'data/chains/' + folder_name + '/shifted_snaked_chains/'
progress_bar = tqdm(total=len(chain_deque), position=0)

for n, chain in enumerate(final_chain_list):
    np.save(dest_folder + 'shifted_snaked_chain_' + str(n) + '.npy', chain)
    progress_bar.update(1)

np.save(dest_folder + 'hist_2nm.npy', hist_2nm)
