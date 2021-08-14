import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% new check dose deposition
def get_E_dep(folder):
    bins = np.arange(1, 2000, 1)
    n_bins = len(bins) - 1

    hist_dE = np.zeros(n_bins)

    bin_centrers = (bins[:-1] + bins[1:])/2

    n_files = 100
    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):

        now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')

        n_electrons = int(np.max(now_data[:, 0]))

        for n_el in range(n_electrons):
            now_e_data = now_data[np.where(now_data[:, 0] == n_el)]
            now_e_z = now_e_data[1:, 6]
            now_e_dE = now_e_data[1:, 7]

            hist_dE += np.histogram(now_e_z, bins=bins, weights=now_e_dE)[0]

        progress_bar.update()

    return bin_centrers, hist_dE


# %%
bin_centrers, hist_dE_100eV = get_E_dep('/Volumes/Transcend/MC_Si/100eV')
bin_centrers, hist_dE_1keV = get_E_dep('/Volumes/Transcend/MC_Si/1keV')
bin_centrers, hist_dE_10keV = get_E_dep('/Volumes/Transcend/MC_Si/10keV')

#%%
bin_centrers, hist_dE_100eV_g = get_E_dep('data/MC_Si_pl/100eV')
# %%
bin_centrers, hist_dE_1keV_g = get_E_dep('data/MC_Si_pl/1000eV')
# %%
bin_centrers, hist_dE_10keV_g = get_E_dep('data/MC_Si_pl/10keV')

# %%
paper_100eV = np.loadtxt('notebooks/Si_distr_check/curves/Si_Edep_100eV.txt')
paper_1keV = np.loadtxt('notebooks/Si_distr_check/curves/Si_Edep_1keV.txt')
paper_10keV = np.loadtxt('notebooks/Si_distr_check/curves/Si_Edep_10keV.txt')

v_10keV = np.loadtxt('notebooks/Si_distr_check/curves/Valentin_2010_10keV.txt')
v_1keV = np.loadtxt('notebooks/Si_distr_check/curves/Valentin_2010_1keV.txt')

plt.figure(dpi=300)

# plt.loglog(v_1keV[:, 0], v_1keV[:, 1], '.', label='old paper 1 keV')
# plt.loglog(v_10keV[:, 0], v_10keV[:, 1], '.', label='old paper 10 keV')

plt.loglog(paper_100eV[:, 0], paper_100eV[:, 1], 'o', label='paper 100eV')
plt.loglog(paper_1keV[:, 0], paper_1keV[:, 1], 'o', label='paper 1keV')
plt.loglog(paper_10keV[:, 0], paper_10keV[:, 1], 'o', label='paper 10keV')

plt.loglog(bin_centrers, hist_dE_100eV / 100 / 100, label='my 100 eV')
plt.loglog(bin_centrers, hist_dE_1keV / 100 / 100, label='my 1 keV')
plt.loglog(bin_centrers, hist_dE_10keV / 100 / 100, label='my 10 keV')

plt.loglog(bin_centrers, hist_dE_100eV_g / 100 / 100, label='my 100 eV geant4')
plt.loglog(bin_centrers, hist_dE_1keV_g / 100 / 100, label='my 1 keV geant4')
plt.loglog(bin_centrers, hist_dE_10keV_g / 100 / 100, label='my 10 keV geant4')

plt.xlim(1e-1, 5e+3)
plt.ylim(1e-2, 1e+3)

plt.xlabel('depth, nm')
plt.ylabel('Dose, eV/nm')

plt.legend()
plt.grid()
# plt.show()
plt.savefig('E_dep.jpg')
# plt.savefig('E_dep_with_2010.jpg')
