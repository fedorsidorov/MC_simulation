import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
n_files = 50
n_bins = 41
hist = np.zeros(n_bins - 1)
bins = np.linspace(0, 4, n_bins)

# %%
progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):
    now_data = np.load('data/MC_Si/10keV/e_DATA_' + str(i) + '.npy')
    e_2nd_arr = now_data[np.where(now_data[:, 8] != 0)[0], 8]
    hist += np.histogram(e_2nd_arr / 1000, bins=bins)[0]
    progress_bar.update()

bin_centrers = (bins[:-1] + bins[1:])/2

# %%
paper_plot = np.loadtxt('notebooks/MC_Si_check/curves/2ndary_spectra.txt')
paper_x = paper_plot[:, 0]
paper_y = paper_plot[:, 1]

plt.figure(dpi=300)

plt.semilogy(paper_x, paper_y * 45000, 'ro', label='paper')
plt.semilogy(bin_centrers, hist, label='my')

plt.xlabel('2ndary e energy, eV')
plt.ylabel('number of 2ndary electrons')

plt.grid()
plt.xlim(0, 4)
plt.legend()

plt.show()
# plt.savefig('2ndary_spectra_new.jpg')

# %%
now_data = np.load('data/MC_Si/10keV_15eV/e_DATA_0.npy')

len(np.where(now_data[:, 7] == 15)[0])


