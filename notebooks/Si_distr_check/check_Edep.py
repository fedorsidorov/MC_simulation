import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% new check dose deposition
# bins = np.arange(1, 2000, 1)
bins = np.arange(0.1, 2000, 1)
n_bins = len(bins) - 1

# hist_dE_100 = np.zeros(n_bins)
# hist_dE_1k = np.zeros(n_bins)
hist_dE_10k = np.zeros(n_bins)

n_e_total = 0

bin_centrers = (bins[:-1] + bins[1:])/2

n_files = 100
n_primaries = 100

progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):
    # now_data = np.load('data/4Akkerman/100eV_pl/e_DATA_' + str(i) + '.npy')
    # now_data = np.load('/Volumes/ELEMENTS/si_si_si/100/e_DATA_' + str(i) + '.npy')

    # now_data = np.load('data/4Akkerman/1keV_pl/e_DATA_' + str(i) + '.npy')
    now_data = np.load('/Volumes/ELEMENTS/si_si_si/1000/e_DATA_' + str(i) + '.npy')

    # now_data = np.load('data/4Akkerman/10keV_pl/e_DATA_' + str(i) + '.npy')
    now_data = np.load('/Volumes/ELEMENTS/si_si_si/10000/e_DATA_' + str(i) + '.npy')

    # now_z = now_data[1:, 6]
    # now_dE = now_data[1:, 7]

    now_z = now_data[1:, 5]
    now_dE = now_data[1:, 6]

    # hist_dE_100 += np.histogram(now_z, bins=bins, weights=now_dE)[0]
    # hist_dE_1k += np.histogram(now_z, bins=bins, weights=now_dE)[0]
    hist_dE_10k += np.histogram(now_z, bins=bins, weights=now_dE)[0]

    progress_bar.update()

# %%
paper_100eV = np.loadtxt('notebooks/Si_distr_check/curves/Si_Edep_100eV.txt')
paper_1keV = np.loadtxt('notebooks/Si_distr_check/curves/Si_Edep_1keV.txt')
paper_10keV = np.loadtxt('notebooks/Si_distr_check/curves/Si_Edep_10keV.txt')

plt.figure(dpi=300)

# plt.loglog(bin_centrers, hist_dE_100 / n_files / n_primaries, label='my 100 eV')
# plt.loglog(bin_centrers, hist_dE_1k / n_files / n_primaries, label='my 1 keV')
plt.loglog(bin_centrers, hist_dE_10k / n_files / n_primaries, label='my 10 keV')

# plt.loglog(paper_100eV[:, 0], paper_100eV[:, 1], 'o', label='paper 100 eV')
# plt.loglog(paper_1keV[:, 0], paper_1keV[:, 1], 'o', label='paper 1 keV')
plt.loglog(paper_10keV[:, 0], paper_10keV[:, 1], 'o', label='paper 10 keV')

# plt.xlim(1e+0, 5e+3)
plt.xlim(1e-1, 5e+3)
plt.ylim(1e-2, 1e+3)

plt.xlabel('depth, nm')
plt.ylabel('dose, eV/nm')

plt.legend()
plt.grid()
plt.show()

# plt.savefig('E_dep_new.jpg')
