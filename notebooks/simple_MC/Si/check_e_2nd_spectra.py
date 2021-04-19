import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def get_2ndary_hist(folder, n_files, n_primaries_in_file, E_2nd_index):
    n_bins = 41
    hist = np.zeros(n_bins - 1)
    bins = np.linspace(0, 4, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):
        now_data = np.load(folder + '/e_DATA_' + str(i) + '.npy')
        E_2nd_arr = now_data[np.where(now_data[:, E_2nd_index] != 0)[0], E_2nd_index]
        hist += np.histogram(E_2nd_arr / 1000, bins=bins)[0]  # E in keVs
        progress_bar.update()

    return bin_centers, hist / n_files / n_primaries_in_file


# %%
# ans = np.load('data/si_si_si/10keV/e_DATA_0.npy')

xx_sim, yy_sim = get_2ndary_hist('data/si_si_si/10000', 100, 100, 7)
# xx_sim_old, yy_sim_old = get_2ndary_hist('/Volumes/Transcend/MC_Si/10keV', 100, 100, 8)

# %%
paper = np.loadtxt('notebooks/MC_Si_check/curves/2ndary_spectra.txt')

plt.figure(dpi=300)

plt.semilogy(paper[:, 0], paper[:, 1] * 10, 'ro', label='paper')
plt.semilogy(xx_sim, yy_sim, label='sim')
# plt.semilogy(xx_sim_old, yy_sim_old, label='sim old')

plt.xlabel('E$_{2nd}$, keV')
plt.ylabel('number of 2ndary electrons')

plt.grid()
plt.xlim(0, 4)
plt.ylim(1e-4, 1e+3)
plt.legend()

# plt.show()
plt.savefig('2ndary_spectra_my_and_geant4.jpg')

# %%
ans = np.load('data/si_si_si/10keV/e_DATA_0.npy')
bns = np.load('/Volumes/Transcend/MC_Si_pl/10keV/e_DATA_0.npy')


