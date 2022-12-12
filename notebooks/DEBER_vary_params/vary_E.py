import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
xx_bins = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/366_vary_params/'

zz_bins_19_sum = np.zeros(len(xx_bins))
zz_bins_21_sum = np.zeros(len(xx_bins))

n_tries = 10

for n_try in range(n_tries):
    # zz_bins_19_sum += np.load(path + 'E_19/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_19_sum += np.load(path + 'E_19p5/try_' + str(n_try) + '/zz_vac_bins.npy')
    # zz_bins_21_sum += np.load(path + 'E_21/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_21_sum += np.load(path + 'E_20p5/try_' + str(n_try) + '/zz_vac_bins.npy')

zz_bins_19_avg = zz_bins_19_sum / n_tries
zz_bins_21_avg = zz_bins_21_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])
# plt.figure(dpi=600)

# plt.plot(xx_bins, zz_bins_19_avg, label=r'19 кэВ')
plt.plot(xx_bins, zz_bins_19_avg, label=r'19.5 кэВ')
# plt.plot(xx_bins, zz_bins_21_avg, label=r'21 кэВ')
plt.plot(xx_bins, zz_bins_21_avg, label=r'20.5 кэВ')

plt.plot(xx_bins, zz_REF - 2, 'k--', label=r'20 кэВ')
plt.plot(xx_bins, zz_REF + 2, 'k--')

# plt.plot(xx_bins, zz_bins_21_sum - zz_bins_19_avg)

plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
plt.ylim(50, 350)

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='lower right')

plt.grid()

plt.savefig('vary_E_1.jpg', dpi=600, bbox_inches='tight')
plt.show()




