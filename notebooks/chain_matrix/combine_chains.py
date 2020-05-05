import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
import constants_mapping as const_m
import constants_physics as const_p

cf = importlib.reload(cf)
af = importlib.reload(af)
pf = importlib.reload(pf)
const_m = importlib.reload(const_m)
const_p = importlib.reload(const_p)


# %%
def get_shift(x_bin, y_bin, z_bin):
    shift = [
        (const_m.x_bins_2nm[x_bin] + const_m.x_bins_2nm[x_bin + 1]) / 2,
        (const_m.y_bins_2nm[y_bin] + const_m.y_bins_2nm[y_bin + 1]) / 2,
        (const_m.z_bins_2nm[z_bin] + const_m.z_bins_2nm[z_bin + 1]) / 2
    ]
    return shift


def get_window_zero_shift(array, wind_size):
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
                    min_pos = i, j, k

    return get_shift(*min_pos)


def get_single_zero_shift(array):
    inds_xyz = np.array(np.where(array == 0)).transpose()
    xyz_bin = inds_xyz[np.random.choice(len(inds_xyz))]
    return get_shift(*xyz_bin)


# %%
chain_lens = np.load('data/Harris/prepared_chain_lens_array.npy')
chain_list = []

for n, _ in enumerate(chain_lens):
    chain_list.append(np.load('data/Harris/prepared_chains/prepared_chain_' + str(n) + '.npy'))

# %%
lens_unique = np.zeros((len(np.unique(chain_lens)), 2))

for n, len_unique in enumerate(np.unique(chain_lens)):
    lens_unique[n, :] = len_unique, len(np.where(chain_lens == len_unique)[0])

# %%
hist_2nm = np.zeros(const_m.hist_2nm_shape)

final_chain_list = []
chain_lens_list = []
n_monomers_now = 0

progress_bar = tqdm(total=len(chain_list), position=0)

for now_chain in chain_list:

    now_chain -= now_chain[0, :]

    now_chain_l_xyz = np.max(now_chain, axis=0) - np.min(now_chain, axis=0)

    # now_shift = np.random.uniform(const_m.xyz_min, const_m.xyz_max)
    # now_shift = get_single_zero_shift(hist_2nm)

    # window_size = int(np.min(now_chain_l_xyz) // const_m.step_2nm)
    window_size = 5
    now_shift = get_window_zero_shift(hist_2nm, window_size)

    now_chain_shifted = now_chain + now_shift
    af.snake_array(now_chain_shifted, 0, 1, 2, const_m.xyz_min, const_m.xyz_max)
    final_chain_list.append(now_chain_shifted)

    hist_2nm += np.histogramdd(now_chain_shifted, bins=const_m.bins_2nm)[0]

    n_monomers_now = np.sum(hist_2nm)
    progress_bar.update()

# %%
plt.imshow(np.average(hist_2nm, axis=1))
plt.show()

# %%
print('2nm average =', np.average(hist_2nm))
print('2nm average density =', np.average(hist_2nm) * const_p.m_MMA / 2e-7 ** 3)

# %%
n_empty = 50 * 50 * 250 - np.count_nonzero(hist_2nm)
part_empty = n_empty / (50 * 50 * 250)

print(part_empty)

# %% save chains to files
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(chain_list):
    np.save('data/Harris/Chains/chain_shifted_snaked_' + str(n) + '.npy', chain)
    progress_bar.update(1)

# %%
np.save('data/Harris/chain_lens_array.npy', chain_lens_array)
np.save('data/Harris/hist_2nm.npy', hist_2nm)


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

# %%
hist_2nm = np.load('data/Harris/hist_2nm.npy')
chain_lens_array = np.load('data/Harris/chain_lens_array.npy')
