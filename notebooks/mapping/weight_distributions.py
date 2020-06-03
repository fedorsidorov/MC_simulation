import importlib

import matplotlib.pyplot as plt
import numpy as np

import constants as cp

cp = importlib.reload(cp)

# %%
deg_paths = 'C-C2:4_C-C\':2'

lens_initial = np.load('data/choi_weight/harris_lens_initial.npy')
# lens_initial = np.load('/Volumes/ELEMENTS/PyCharm_may/prepared_chains/Harris/chain_lens.npy')
# lens_final = np.load('data/Harris/lens_final_' + deg_path + '.npy')
lens_final = np.load('data/exposed_chains/Harris/harris_lens_final_4+2_2nm.npy')

mass_initial = lens_initial * 100
mass_final = lens_final * cp.u_MMA

# harris_mass = np.load('Resources/Harris/harris_x_before.npy')
# harris_distribution = np.load('Resources/Harris/harris_y_before_fit.npy')

# %%
# bins = np.logspace(2, 7.1, 21)
bins = np.logspace(0, 7.1, 41)
bin_centers = (bins[:-1] + bins[1:]) / 2
hist_initial = np.histogram(mass_initial, bins, normed=True)[0]
hist_final = np.histogram(mass_final, bins, normed=True)[0]

plt.figure(dpi=300)
plt.semilogx(bin_centers, hist_initial / np.max(hist_initial), 'o-', label='initial')
plt.semilogx(bin_centers, hist_final / np.max(hist_final), 'o-', label='final')

# plt.gca().set_xscale('log')
plt.xlabel('molecular weight')
plt.ylabel('N$_{entries}$')
plt.xlim(1e+0, 1e+8)
plt.legend()
plt.grid()
plt.show()
