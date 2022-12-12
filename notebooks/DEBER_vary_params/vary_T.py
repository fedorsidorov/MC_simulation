import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
xx_bins = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/366_vary_params/'

zz_bins_147_sum = np.zeros(len(xx_bins))
zz_bins_148_sum = np.zeros(len(xx_bins))
zz_bins_149_sum = np.zeros(len(xx_bins))
zz_bins_151_sum = np.zeros(len(xx_bins))
zz_bins_152_sum = np.zeros(len(xx_bins))
zz_bins_153_sum = np.zeros(len(xx_bins))

n_tries = 10

for n_try in range(n_tries):
    zz_bins_147_sum += np.load(path + 'T_147/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_148_sum += np.load(path + 'T_148/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_149_sum += np.load(path + 'T_149/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_151_sum += np.load(path + 'T_151/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_152_sum += np.load(path + 'T_152/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_153_sum += np.load(path + 'T_153/try_' + str(n_try) + '/zz_vac_bins.npy')

zz_bins_147_avg = zz_bins_147_sum / n_tries
zz_bins_148_avg = zz_bins_148_sum / n_tries
zz_bins_149_avg = zz_bins_149_sum / n_tries
zz_bins_151_avg = zz_bins_151_sum / n_tries
zz_bins_152_avg = zz_bins_152_sum / n_tries
zz_bins_153_avg = zz_bins_153_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])

plt.plot([0], [0], 'k--', label=r'$150^\circ$C')

# plt.plot(xx_bins, zz_bins_147_avg, label=r'$147^\circ$C')
# plt.plot(xx_bins, zz_bins_148_avg, label=r'$148^\circ$C')
plt.plot(xx_bins, zz_bins_149_avg, label=r'$149^\circ$C')
# plt.plot(xx_bins, zz_bins_151_avg, label=r'$151^\circ$C')
# plt.plot(xx_bins, zz_bins_152_avg, label=r'$152^\circ$C')
plt.plot(xx_bins, zz_bins_153_avg, label=r'$153^\circ$C')

plt.plot(xx_bins, zz_REF - 2, 'k--')
plt.plot(xx_bins, zz_REF + 2, 'k--')

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='lower right')

plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
plt.ylim(50, 350)
plt.grid()

# plt.savefig('vary_T_center.jpg', dpi=600, bbox_inches='tight')
plt.show()




