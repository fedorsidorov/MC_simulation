import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% check range
def get_true_proj_ranges(folder, n_files, n_primaries):

    total_range = 0
    total_range_z = 0

    for n in range(n_files):

        data = np.load(folder + 'e_DATA_' + str(n) + '.npy')

        for n_pr in range(n_primaries):

            now_data = data[np.where(data[:, 0] == n_pr)]

            now_e_coords = now_data[:, 4:7]
            now_e_z = now_data[:, 6]

            total_range += np.sum(np.linalg.norm(now_e_coords[1:, :] - now_e_coords[:-1, :], axis=1))
            total_range_z += np.sum(np.abs(now_e_z[1:] - now_e_z[:-1]))

    return total_range / n_primaries / n_files, total_range_z / n_primaries / n_files


# %%
# paper_range = np.loadtxt('notebooks/Si_distr_check/curves/Si_true_range.txt')
paper_range = np.loadtxt('notebooks/Si_distr_check/curves_2010/Si_range_2010.txt')

# paper_range_z = np.loadtxt('notebooks/Si_distr_check/curves/Si_projected_range.txt')
# paper_range_z_livermore = np.loadtxt('notebooks/Akkerman_Si_5osc/curves/Si_Livermore.txt')

# %%
n_primaries = 100
EE = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000]
nn_files = [300, 300, 300, 300, 300, 100, 100, 100, 100, 100]

true_ranges = np.zeros(len(EE))
proj_ranges = np.zeros(len(EE))

progress_bar = tqdm(total=len(EE), position=0)

for i in range(len(EE)):
    true_ranges[i] = get_true_proj_ranges('data/4Akkerman/' + str(EE[i]) + '/', nn_files[i], n_primaries)[0]
    proj_ranges[i] = get_true_proj_ranges('data/4Akkerman/' + str(EE[i]) + '/', nn_files[i], n_primaries)[1]

    progress_bar.update()

# %%
plt.figure(dpi=300)

plt.loglog(paper_range[:, 0], paper_range[:, 1], 'o-', label='paper ranges')
plt.loglog(EE, true_ranges, '*-', label='my ranges')

# plt.loglog(paper_range_z[:, 0], paper_range_z[:, 1], 'o-', label='paper z rangez')
# plt.loglog(paper_range_z_livermore[:, 0], paper_range_z_livermore[:, 1], 'o-', label='paper z rangez Livermore')
# plt.loglog(EE, proj_ranges, '*-', label='my ranges')

plt.xlabel('electron energy, eV')
plt.ylabel('electron range, nm')

plt.legend()
plt.grid()

plt.xlim(1e+2, 3e+4)
plt.ylim(1e+0, 1e+5)

plt.show()
# plt.savefig('true_z_ranges.jpg')
