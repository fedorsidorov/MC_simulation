import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% new check dose deposition
def get_E_dep(folder, n_files, n_primaries_in_file, z_ind, E_dep_ind):
    bins = np.arange(1, 2000, 1)
    n_bins = len(bins) - 1

    hist_dE = np.zeros(n_bins)
    bin_centrers = (bins[:-1] + bins[1:])/2

    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):

        now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')
        n_electrons = int(np.max(now_data[:, 0]))

        for n_el in range(n_electrons):
            now_e_data = now_data[np.where(now_data[:, 0] == n_el)]
            now_e_z = now_e_data[1:, z_ind]
            now_e_dE = now_e_data[1:, E_dep_ind]
            hist_dE += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]

        progress_bar.update()

    return bin_centrers, hist_dE / n_files / n_primaries_in_file


# %%
xx_sim_100eV, yy_sim_100eV = get_E_dep('data/si_si_si/100', 100, 100, z_ind=5, E_dep_ind=6)
xx_sim_1keV, yy_sim_1keV = get_E_dep('data/si_si_si/1000', 100, 100, z_ind=5, E_dep_ind=6)
xx_sim_10keV, yy_sim_10keV = get_E_dep('data/si_si_si/10000', 100, 100, z_ind=5, E_dep_ind=6)

# xx_sim_old, yy_sim_old = get_E_dep('/Volumes/Transcend/MC_Si/10keV', n_files=100, z_ind=6, E_dep_ind=7)

# %%
paper_100eV = np.loadtxt('notebooks/MC_Si_check/curves/Si_Edep_100eV.txt')
paper_1keV = np.loadtxt('notebooks/MC_Si_check/curves/Si_Edep_1keV.txt')
paper_10keV = np.loadtxt('notebooks/MC_Si_check/curves/Si_Edep_10keV.txt')

# paper_10keV_old = np.loadtxt('notebooks/MC_Si_check/curves/Valentin_2010_10keV.txt')

plt.figure(dpi=300)

plt.loglog(paper_100eV[:, 0], paper_100eV[:, 1], 'o', label='paper 100 eV')
plt.loglog(paper_1keV[:, 0], paper_1keV[:, 1], 'o', label='paper 1 keV')
plt.loglog(paper_10keV[:, 0], paper_10keV[:, 1], 'o', label='paper 10 keV')

# plt.loglog(paper_10keV_old[:, 0], paper_10keV_old[:, 1], '.', label='paper 10keV щдв')

plt.loglog(xx_sim_100eV, yy_sim_100eV, label='my 100 eV')
plt.loglog(xx_sim_1keV, yy_sim_1keV, label='my 1 keV')
plt.loglog(xx_sim_10keV, yy_sim_10keV, label='my 10 keV')
# plt.loglog(xx_sim_old, yy_sim_old, label='my 10 keV')

plt.xlim(1e+0, 5e+3)
plt.ylim(1e-2, 1e+3)

plt.xlabel('depth, nm')
plt.ylabel('Dose, eV/nm')

plt.legend()
plt.grid()
# plt.show()
plt.savefig('E_dep.jpg')

# %%
ans = np.load('data/si_si_si/10000/e_DATA_0.npy')
# bns = np.load('/Volumes/Transcend/MC_Si/10keV/e_DATA_0.npy')
