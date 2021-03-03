import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% check range
def get_true_proj_ranges(folder, n_files, n_primaries):

    total_range = 0
    total_range_z = 0

    for n in range(n_files):

        data = np.load(folder + '/e_DATA_' + str(n) + '.npy')

        for n_pr in range(n_primaries):

            now_data = data[np.where(data[:, 0] == n_pr)]

            now_e_coords = now_data[:, 4:7]
            now_e_z = now_data[:, 6]

            total_range += np.sum(np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1))
            total_range_z += np.sum(np.abs(now_e_z[1:] - now_e_z[:-1]))

    return total_range / n_primaries / n_files, total_range_z / n_primaries / n_files


# %%
paper_range = np.loadtxt('notebooks/MC_Si_check/curves/Si_true_range.txt')
paper_range_z = np.loadtxt('notebooks/MC_Si_check/curves/Si_projected_range.txt')

true_range_100, z_range_100 = get_true_proj_ranges('data/MC_Si/100eV', 100, 100)
true_range_200, z_range_200 = get_true_proj_ranges('data/MC_Si/200eV', 100, 100)
true_range_300, z_range_300 = get_true_proj_ranges('data/MC_Si/300eV', 100, 100)

true_range_1k, z_range_1k = get_true_proj_ranges('data/MC_Si/1keV', 100, 100)
true_range_10k, z_range_10k = get_true_proj_ranges('data/MC_Si/10keV', 100, 100)

# %%
true_range_400, z_range_400 = get_true_proj_ranges('data/MC_Si/400eV', 100, 100)
# true_range_500, z_range_500 = get_true_proj_ranges('data/MC_Si/500eV', 100, 100)
# true_range_600, z_range_600 = get_true_proj_ranges('data/MC_Si/600eV', 100, 100)

plt.figure(dpi=300)

plt.loglog(paper_range[:, 0], paper_range[:, 1], label='paper range')
plt.loglog(paper_range_z[:, 0], paper_range_z[:, 1], label='paper z range')

plt.loglog(
    [100, 200, 300, 400, 1000, 10000],
    [true_range_100, true_range_200, true_range_300,  true_range_400, true_range_1k, true_range_10k],
    'o', label='true range')

plt.loglog(
    [100, 200, 300, 400, 1000, 10000],
    [z_range_100, z_range_200, z_range_300, z_range_400, z_range_1k, z_range_10k],
    'o', label='true range')

plt.xlabel('electron energy, eV')
plt.ylabel('electron range, nm')

plt.legend()
plt.grid()

plt.show()
# plt.savefig('projected_range_new.jpg')
