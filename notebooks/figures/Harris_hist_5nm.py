import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
from mapping import mapping_harris as mapping
import constants as const

mapping = importlib.reload(mapping)
const = importlib.reload(const)
cf = importlib.reload(cf)
af = importlib.reload(af)
pf = importlib.reload(pf)


# %%
def get_shift(x_bin, y_bin, z_bin):
    shift = [
        (mapping.x_bins_5nm[x_bin] + mapping.x_bins_5nm[x_bin + 1]) / 2,
        (mapping.y_bins_5nm[y_bin] + mapping.y_bins_5nm[y_bin + 1]) / 2,
        (mapping.z_bins_5nm[z_bin] + mapping.z_bins_5nm[z_bin + 1]) / 2
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
    inds_xyz = np.array(np.where(array == array.min())).transpose()
    xyz_bin = inds_xyz[np.random.choice(len(inds_xyz))]
    return get_shift(*xyz_bin)


# %%
source_folder = '/Volumes/TOSHIBA EXT/chains_harris/prepared_chains_3/'

chain_lens = np.load(source_folder + 'chain_lens.npy')
chain_list = []

progress_bar = tqdm(total=len(chain_lens), position=0)

ends_diff_array = np.zeros((len(chain_lens), 3))

for n, _ in enumerate(chain_lens):
    now_chain = np.load(source_folder + 'chain_' + str(n) + '.npy')
    ends_diff_array[n, :] = (now_chain[-1, :] - now_chain[0, :]) ** 2
    step_array = np.linalg.norm((now_chain[1:, :] - now_chain[:-1, :]), axis=1)
    chain_list.append(now_chain)
    progress_bar.update()

# %%
np.average(ends_diff_array)

# %%
final_chain_list = []
chain_lens_list = []
n_monomers_now = 0

progress_bar = tqdm(total=len(chain_list), position=0)

hist_5nm = np.zeros(mapping.hist_5nm_shape)

for n, now_chain in enumerate(chain_list):

    now_chain -= now_chain[0, :]
    now_chain_center_xyz = (np.max(now_chain, axis=0) + np.min(now_chain, axis=0)) / 2
    now_chain = now_chain - now_chain_center_xyz

    a, b, g = np.random.random(3) * 2 * np.pi
    now_chain = cf.rotate_chain(now_chain, a, b, g)

    now_shift = get_zero_shift(hist_5nm)
    now_chain_shifted = now_chain + now_shift

    af.snake_array(now_chain_shifted, 0, 1, 2, mapping.xyz_min, mapping.xyz_max)
    final_chain_list.append(now_chain_shifted)

    hist_5nm += np.histogramdd(now_chain_shifted, bins=mapping.bins_5nm)[0]

    n_monomers_now = np.sum(hist_5nm)

    progress_bar.update()


# %%
plt.figure(dpi=600, figsize=[2.5, 5.5])
plt.imshow(np.average(hist_5nm, axis=0).transpose(), extent=[-50, 50, 0, 500])

plt.colorbar()

plt.xlabel('x, нм')
plt.ylabel('z, нм')

plt.savefig('figures/Harris_hist_5nm.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %%
# plt.imshow(np.average(hist_5nm, axis=0))
# plt.show()

with plt.style.context(['science', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    plt.imshow(np.average(hist_5nm, axis=0).transpose(), extent=[-50, 50, 0, 500])
    plt.colorbar()
    ax.set(xlabel='x, nm')
    ax.set(ylabel='z, nm')

    plt.show()
    # fig.savefig('figures/Harris_chains_hist_5nm.jpg', dpi=600)


# %%
n_empty = np.prod(mapping.hist_5nm_shape) - np.count_nonzero(hist_5nm)
part_empty = n_empty / np.prod(mapping.hist_5nm_shape)

print(np.sqrt(np.var(hist_5nm)))
print(part_empty)

# %%
np.save('data/harris_monomer_hist_5nm.npy', hist_5nm)

# %% save chains to files
dest_folder = '/Volumes/ELEMENTS/chains_harris/rot_sh_sn_chains_3/'
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(final_chain_list):
    np.save(dest_folder + 'rot_sh_sn_chain_' + str(n) + '.npy', chain)
    progress_bar.update()

np.save(dest_folder + 'hist_5nm.npy', hist_5nm)
