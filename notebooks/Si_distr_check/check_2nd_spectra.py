import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def get_2ndary_hist(folder, n_files, n_primaries):
    n_bins = 41
    hist = np.zeros(n_bins - 1)
    bins = np.linspace(0, 4, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    keV = 1000  # eV

    progress_bar = tqdm(total=n_files, position=0)

    for i in range(n_files):
        now_data = np.load(folder + 'e_DATA_' + str(i) + '.npy')
        e_2nd_arr = now_data[np.where(now_data[:, 8] != 0)[0], 8]
        hist += np.histogram(e_2nd_arr / keV, bins=bins)[0]
        progress_bar.update()

    return bin_centers, hist / n_files / n_primaries


# %%
n_files = 100
n_primaries = 100

bin_centers, hist_my = get_2ndary_hist('data/4Akkerman/10000/', n_files, n_primaries)

# %%
paper_plot = np.loadtxt('notebooks/Si_distr_check/curves/2ndary_spectra.txt')
paper_x = paper_plot[:, 0]
paper_y = paper_plot[:, 1]

plt.figure(dpi=300)

plt.semilogy(paper_x, paper_y, 'ro', label='paper')
plt.semilogy(bin_centers, hist_my / 10, '.--', label='my')

plt.xlabel('E$_{2nd}$, eV')
plt.ylabel('number of 2ndary electrons')

plt.grid()
plt.xlim(0, 4)
plt.ylim(1e-4, 1e+3)
plt.legend()

plt.show()
# plt.savefig('2ndary_spectra_my.jpg')
