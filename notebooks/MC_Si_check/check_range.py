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
paper_range_z_livermore = np.loadtxt('notebooks/OLF_Si/curves/Si_Livermore.txt')

# %%
true_range_100eV_g, z_range_100eV_g = get_true_proj_ranges('data/MC_Si_pl/100eV', 100, 100)
true_range_400eV_g, z_range_400eV_g = get_true_proj_ranges('data/MC_Si_pl/400eV', 100, 100)
true_range_1keV_g, z_range_1keV_g = get_true_proj_ranges('data/MC_Si_pl/1000eV', 100, 100)
true_range_4keV_g, z_range_4keV_g = get_true_proj_ranges('data/MC_Si_pl/4000eV', 100, 100)
true_range_10keV_g, z_range_10keV_g = get_true_proj_ranges('data/MC_Si_pl/10keV', 100, 100)

true_range_100, z_range_100 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/100eV', 100, 100)
true_range_200, z_range_200 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/200eV', 100, 100)
true_range_300, z_range_300 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/300eV', 100, 100)
true_range_500, z_range_500 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/500eV', 100, 100)
true_range_700, z_range_700 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/700eV', 100, 100)

true_range_1000, z_range_1000 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/1keV', 100, 100)
true_range_2000, z_range_2000 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/2keV', 100, 100)
true_range_3000, z_range_3000 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/3keV', 100, 100)
true_range_5000, z_range_5000 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/5keV', 100, 100)
true_range_7000, z_range_7000 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/7keV', 100, 100)
true_range_10000, z_range_10000 = get_true_proj_ranges('/Volumes/Transcend/MC_Si/10keV', 100, 100)

# %%
# true_range_100_g, z_range_100_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/100eV', 100, 100)
# true_range_200_g, z_range_200_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/200eV', 100, 100)
# true_range_300_g, z_range_300_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/300eV', 100, 100)
# true_range_500_g, z_range_500_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/500eV', 100, 100)
# true_range_700_g, z_range_700_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/700eV', 100, 100)
#
# true_range_1000_g, z_range_1000_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/1keV', 100, 100)
# true_range_2000_g, z_range_2000_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/2keV', 100, 100)
# true_range_3000_g, z_range_3000_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/3keV', 100, 100)
# true_range_5000_g, z_range_5000_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/5keV', 100, 100)
# true_range_7000_g, z_range_7000_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/7keV', 100, 100)
# true_range_10000_g, z_range_10000_g = get_true_proj_ranges('/Volumes/Transcend/MC_Si_geant4/10keV', 100, 100)

# %%
plt.figure(dpi=300)

plt.loglog(paper_range[:, 0], paper_range[:, 1], 'o-', label='paper range')
plt.loglog(paper_range_z[:, 0], paper_range_z[:, 1], 'o-', label='paper z range')
plt.loglog(paper_range_z_livermore[:, 0], paper_range_z_livermore[:, 1], 'o-', label='paper z range Livermore')

plt.loglog(
    [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000],
    [true_range_100, true_range_200, true_range_300,  true_range_500, true_range_700,
     true_range_1000, true_range_2000, true_range_3000, true_range_5000, true_range_7000, true_range_10000],
    'b*', label='true range')

plt.loglog(
    [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000],
    [z_range_100, z_range_200, z_range_300,  z_range_500, z_range_700,
     z_range_1000, z_range_2000, z_range_3000, z_range_5000, z_range_7000, z_range_10000],
    'r*', label='z range')

plt.loglog(
    [100, 400, 1000, 4000, 10000],
    [true_range_100eV_g, true_range_400eV_g, true_range_1keV_g, true_range_4keV_g, true_range_10keV_g],
    # [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000],
    # [true_range_100_g, true_range_200_g, true_range_300_g,  true_range_500_g, true_range_700_g,
    #  true_range_1000_g, true_range_2000_g, true_range_3000_g, true_range_5000_g, true_range_7000_g, true_range_10000_g],
    'b^', label='true range G')

plt.loglog(
    [100, 400, 1000, 4000, 10000],
    [z_range_100eV_g, z_range_400eV_g, z_range_1keV_g, true_range_4keV_g, z_range_10keV_g],
    # [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000],
    # [z_range_100_g, z_range_200_g, z_range_300_g,  z_range_500_g, z_range_700_g,
    #  z_range_1000_g, z_range_2000_g, z_range_3000_g, z_range_5000_g, z_range_7000_g, z_range_10000_g],
    'r^', label='z range G')

plt.xlabel('electron energy, eV')
plt.ylabel('electron range, nm')

plt.legend()
plt.grid()

plt.show()
# plt.savefig('true_z_ranges.jpg')
