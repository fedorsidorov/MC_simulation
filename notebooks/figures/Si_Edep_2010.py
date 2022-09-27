import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% new check dose deposition
# bins = np.arange(1, 2000, 1)
bins = np.arange(0.1, 2000, 1)
n_bins = len(bins) - 1

hist_dE_250 = np.zeros(n_bins)
hist_dE_1k = np.zeros(n_bins)
hist_dE_10k = np.zeros(n_bins)

n_e_total = 0

bin_centrers = (bins[:-1] + bins[1:])/2

n_files = 100
n_primaries = 100

progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):

    now_data_250 = np.load('/Volumes/Transcend/4Akkerman/250/e_DATA_' + str(i) + '.npy')
    now_data_1k = np.load('/Volumes/Transcend/4Akkerman/1000/e_DATA_' + str(i) + '.npy')
    now_data_10k = np.load('/Volumes/Transcend/4Akkerman/10000/e_DATA_' + str(i) + '.npy')

    now_z_250 = now_data_250[1:, 6]
    now_dE_250 = now_data_250[1:, 7]

    now_z_1k = now_data_1k[1:, 6]
    now_dE_1k = now_data_1k[1:, 7]

    now_z_10k = now_data_10k[1:, 6]
    now_dE_10k = now_data_10k[1:, 7]

    hist_dE_250 += np.histogram(now_z_250, bins=bins, weights=now_dE_250)[0]
    hist_dE_1k += np.histogram(now_z_1k, bins=bins, weights=now_dE_1k)[0]
    hist_dE_10k += np.histogram(now_z_10k, bins=bins, weights=now_dE_10k)[0]

    progress_bar.update()

# %%
paper_250 = np.loadtxt('notebooks/Si_distr_check/curves_2010/Si_dose_250eV_2010_NEW.txt')
paper_1k = np.loadtxt('notebooks/Si_distr_check/curves_2010/Si_dose_1keV_2010_NEW.txt')
paper_10k = np.loadtxt('notebooks/Si_distr_check/curves_2010/Si_dose_10keV_2010_NEW.txt')

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    plt.loglog(paper_250[:, 0], paper_250[:, 1], '.', label='250 eV')
    plt.loglog(paper_1k[:, 0], paper_1k[:, 1], '.', label='1 keV')
    plt.loglog(paper_10k[:, 0], paper_10k[:, 1], '.', label='10 keV')

    plt.loglog(bin_centrers, hist_dE_250 / n_files / n_primaries, label='мое 100 eV')
    plt.loglog(bin_centrers, hist_dE_1k / n_files / n_primaries, label='мое 1 keV')
    plt.loglog(bin_centrers, hist_dE_10k / n_files / n_primaries, label='мое 10 keV')

    ax.legend(loc=1, fontsize=7)
    ax.set(xlabel=r'глубина, нм')
    ax.set(ylabel=r'доза, эВ/нм')
    ax.autoscale(tight=True)

    plt.xlim(1e-1, 1e+4)
    plt.ylim(1e-2, 1e+4)

    # plt.show()
    fig.savefig('figures/Si_spectra_2010.jpg', dpi=600)
