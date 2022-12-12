import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
xx_bins = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/366_vary_params/'

zz_bins_1p1_sum = np.zeros(len(xx_bins))
zz_bins_1p3_sum = np.zeros(len(xx_bins))

n_tries = 10

for n_try in range(n_tries):
    # zz_bins_1p1_sum += np.load(path + 'I_1p15/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_1p1_sum += np.load(path + 'I_1p18/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_1p3_sum += np.load(path + 'I_1p25/try_' + str(n_try) + '/zz_vac_bins.npy')

zz_bins_1p1_avg = zz_bins_1p1_sum / n_tries
zz_bins_1p3_avg = zz_bins_1p3_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])
# plt.figure(dpi=600)

# plt.plot([0], [0], 'k--', label=r'1.2 нА')
plt.plot([0], [0], 'k--', label=r'4.6 нА')

# plt.plot(xx_bins, zz_bins_1p1_avg, label=r'1.15 нА')

# plt.plot(xx_bins, zz_bins_1p1_avg, label=r'1.18 нА')
plt.plot(xx_bins, zz_bins_1p1_avg, label=r'4.5 нА')

# plt.plot(xx_bins, zz_bins_1p3_avg, label=r'1.25 нА')
plt.plot(xx_bins, zz_bins_1p3_avg, label=r'1.7 нА')

plt.plot(xx_bins, zz_REF - 2, 'k--')
plt.plot(xx_bins, zz_REF + 2, 'k--')

plt.xlim(-1500, 1500)
plt.ylim(50, 350)

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='lower right')

plt.grid()

# plt.savefig('vary_I.jpg', dpi=600, bbox_inches='tight')
plt.show()




