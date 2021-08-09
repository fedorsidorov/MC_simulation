import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% check range
def get_true_proj_ranges(folder, n_files, n_primaries):

    total_range = 0
    total_range_z = 0

    progress_bar = tqdm(total=n_files, position=0)

    for n in range(n_files):

        data = np.load(folder + '/e_DATA_' + str(n) + '.npy')

        for n_pr in range(n_primaries):

            now_data = data[np.where(data[:, 0] == n_pr)]

            now_e_coords = now_data[:, 3:6]
            now_e_z = now_data[:, 5]

            total_range += np.sum(np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1))
            total_range_z += np.sum(np.abs(now_e_z[1:] - now_e_z[:-1]))

        progress_bar.update()

    return total_range / n_primaries / n_files, total_range_z / n_primaries / n_files


# %%
paper_range = np.loadtxt('notebooks/Si_distr_check/curves/Si_true_range.txt')
paper_range_z = np.loadtxt('notebooks/Si_distr_check/curves/Si_projected_range.txt')
paper_range_z_livermore = np.loadtxt('notebooks/OLF_Si/curves/Si_Livermore.txt')

# %%
# E0_arr = [20, 40, 50, 60, 70, 80, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000]
# n_elec = [5000, 2000, 5000, 2000, 2000, 2000, 2000, 2000, 2000, 100, 100, 100, 100, 100, 100]

E0_arr = [20, 40, 60, 80, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000]
n_elec = [5000, 2000, 2000, 2000, 2000, 2000, 2000, 100, 100, 100, 100, 100, 100]

true_ranges = []
z_ranges = []

for i, E0 in enumerate(E0_arr):
    print(E0)
    true_r, z_r = get_true_proj_ranges('data/si_si_si/' + str(E0), n_elec[i], 100)
    true_ranges.append(true_r)
    z_ranges.append(z_r)

# %%
plt.figure(dpi=300)

# plt.loglog(paper_range[:, 0], paper_range[:, 1], 'o', label='paper range')
plt.loglog(paper_range_z[:, 0], paper_range_z[:, 1], 'o', label='paper z range')
plt.loglog(paper_range_z_livermore[:, 0], paper_range_z_livermore[:, 1], 'o', label='paper z range Livermore')

# plt.loglog(E0_arr, true_ranges, '*-', label='true range sim')
plt.loglog(E0_arr, z_ranges, 'v-', label='z range_sim')

plt.xlabel('electron energy, eV')
plt.ylabel('electron range, nm')

plt.legend()
plt.grid()

plt.xlim(10, 2e+4)

# plt.show()
plt.savefig('z_range.jpg')
