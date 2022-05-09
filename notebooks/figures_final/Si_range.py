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
paper_range = np.loadtxt('notebooks/Si_distr_check/curves/Si_true_range.txt')
paper_range_z = np.loadtxt('notebooks/Si_distr_check/curves/Si_z_range.txt')
paper_range_z_L = np.loadtxt('notebooks/Si_distr_check/curves/Si_z_range_Livermore.txt')

# %%
n_primaries = 100
# EE = [25, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 100, 250]
EE = [25, 45, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000]

true_ranges = np.zeros(len(EE))
proj_ranges = np.zeros(len(EE))

progress_bar = tqdm(total=len(EE), position=0)

for i in range(len(EE)):

    if EE[i] < 1000:
        n_files = 300
    else:
        n_files = 100

    true_ranges[i] = get_true_proj_ranges('/Volumes/Transcend/4Akkerman/' + str(EE[i]) + '/', n_files, n_primaries)[0]
    proj_ranges[i] = get_true_proj_ranges('/Volumes/Transcend/4Akkerman/' + str(EE[i]) + '/', n_files, n_primaries)[1]

    progress_bar.update()

# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)
    ax.loglog(np.concatenate((paper_range[:1, 0], paper_range[2:, 0])), np.concatenate((paper_range[:1, 1], paper_range[2:, 1])), '.--', label='статья')
    ax.loglog(EE, true_ranges, 'r.--', label='мое')

    # plt.loglog(paper_range_z[:, 0], paper_range_z[:, 1], 'o-', label='paper z ranges')
    # plt.loglog(paper_range_z_L[:, 0], paper_range_z_L[:, 1], 'o-', label='paper z ranges Livermore')
    # plt.loglog(EE, proj_ranges, '*-', label='my z ranges')

    ax.set(xlabel=r'энергия электрона, эВ')
    ax.set(ylabel='длина пробега, нм')

    ax.legend(fontsize=7)

    # plt.xlim(1e+2, 3e+4)
    # plt.ylim(1e+0, 1e+5)

    fig.show()

