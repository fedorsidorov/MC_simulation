import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
# import mapping_harris as mapping
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


def get_window_shift(array, wind_size):
    half = wind_size // 2
    len_x, len_y, len_z = np.shape(array)
    min_sum = np.inf
    min_pos = [0, 0, 0]

    for i in range(half, len_x - half):
        for j in range(half, len_y - half):
            for k in range(half, len_z - half):
                now_sum = np.sum(array[i - half:i + half + 1, j - half:j + half + 1, k - half:k + half + 1])
                if now_sum == 0:
                    return get_shift(i, j, k)
                elif now_sum < min_sum:
                    min_sum = now_sum
                    min_pos = i, j, k

    return get_shift(*min_pos)


def get_zero_shift(array):
    inds_xyz = np.array(np.where(array == 0)).transpose()
    xyz_bin = inds_xyz[np.random.choice(len(inds_xyz))]
    return get_shift(*xyz_bin)


# %%
folder_name = 'Aktary'
source_folder = 'data/chains/' + folder_name + '/prepared_chains/'

chain_lens = np.load(source_folder + 'prepared_chain_lens.npy')
chain_list = []

progress_bar = tqdm(total=len(chain_lens), position=0)

for n, _ in enumerate(chain_lens):
    chain_list.append(np.load(source_folder + 'prepared_chain_' + str(n) + '.npy'))
    progress_bar.update()

# %%
hist_2nm = np.zeros(mapping.hist_2nm_shape)

final_chain_list = []
chain_lens_list = []
n_monomers_now = 0

chains_l_xyz_array = np.zeros((len(chain_lens), 3))

progress_bar = tqdm(total=len(chain_list), position=0)

for n, now_chain in enumerate(chain_list):

    now_chain -= now_chain[0, :]
    now_chain_center_xyz = (np.max(now_chain, axis=0) + np.min(now_chain, axis=0)) / 2
    now_chain = now_chain - now_chain_center_xyz

    now_chain_l_xyz = np.max(now_chain, axis=0) - np.min(now_chain, axis=0)
    chains_l_xyz_array[n, :] = now_chain_l_xyz

    # now_shift = np.random.uniform(const_m.xyz_min, const_m.xyz_max)
    # now_shift = get_zero_shift(hist_2nm)

    # window_size = 10
    # window_size = int(np.average(now_chain_l_xyz) // mapping.step_2nm)
    window_size = 5
    now_shift = get_window_shift(hist_2nm, window_size)

    now_chain_shifted = now_chain + now_shift
    af.snake_array(now_chain_shifted, 0, 1, 2, mapping.xyz_min, mapping.xyz_max)
    final_chain_list.append(now_chain_shifted)

    hist_2nm += np.histogramdd(now_chain_shifted, bins=mapping.bins_2nm)[0]

    n_monomers_now = np.sum(hist_2nm)
    progress_bar.update()


# %%
plt.imshow(np.average(hist_2nm, axis=0))
plt.show()

# %%
n_empty = np.prod(mapping.hist_2nm_shape) - np.count_nonzero(hist_2nm)
part_empty = n_empty / np.prod(mapping.hist_2nm_shape)

print(np.sqrt(np.var(hist_2nm)))
print(part_empty)

# %% save chains to files
# data/chains/Aktary/shifted_snaked_chains
dest_folder = 'data/chains/' + folder_name + '/shifted_snaked_chains/'
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(final_chain_list):
    np.save(dest_folder + 'shifted_snaked_chain_' + str(n) + '.npy', chain)
    progress_bar.update(1)

np.save(dest_folder + 'hist_2nm.npy', hist_2nm)
