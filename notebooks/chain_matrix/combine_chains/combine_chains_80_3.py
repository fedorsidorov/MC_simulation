import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
import mapping_exp_80_3 as mapping
import constants as const

cf = importlib.reload(cf)
af = importlib.reload(af)
pf = importlib.reload(pf)
mapping = importlib.reload(mapping)
const = importlib.reload(const)


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
source_folder = '/Volumes/ELEMENTS/chains_950K/prepared_chains/exp_80_3/'

chain_lens = np.load(source_folder + 'chain_lens.npy')
chain_list = []

progress_bar = tqdm(total=len(chain_lens), position=0)

diff_array = np.zeros((len(chain_lens), 3))

for n, _ in enumerate(chain_lens):
    now_chain = np.load(source_folder + 'chain_' + str(n) + '.npy')
    # diff_array[n, :] = now_chain[-1, :] - now_chain[0, :]
    # diff_array[n, :] = np.abs(now_chain[-1, :] - now_chain[0, :])
    diff_array[n, :] = (now_chain[-1, :] - now_chain[0, :])**2
    chain_list.append(np.load(source_folder + 'chain_' + str(n) + '.npy'))
    progress_bar.update()

# %%
np.average(diff_array)

# %%
final_chain_list = []
chain_lens_list = []
n_monomers_now = 0
hist_2nm = np.zeros(mapping.hist_2nm_shape)

progress_bar = tqdm(total=len(chain_list), position=0)

for n, now_chain in enumerate(chain_list):

    now_chain -= now_chain[0, :]
    now_chain_center_xyz = (np.max(now_chain, axis=0) + np.min(now_chain, axis=0)) / 2
    now_chain = now_chain - now_chain_center_xyz

    a, b, g = np.random.random(3) * 2 * np.pi
    now_chain = cf.rotate_chain(now_chain, a, b, g)

    # now_chain_l_xyz = np.max(now_chain, axis=0) - np.min(now_chain, axis=0)
    now_shift = get_zero_shift(hist_2nm)
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
dest_folder = '/Volumes/ELEMENTS/chains_950K/rot_sh_sn_chains/exp_80_3/'
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(final_chain_list):
    np.save(dest_folder + 'rot_sh_sn_chain_' + str(n) + '.npy', chain)
    progress_bar.update()

np.save(dest_folder + 'hist_2nm.npy', hist_2nm)
