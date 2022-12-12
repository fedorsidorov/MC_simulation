import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)

xx_REF = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/150C_resolution/'


# %% 5 keV
xx_bins = np.load(path + '5_keV/10C_sec/40/xx_bins.npy')
xx_total = np.load(path + '5_keV/10C_sec/40/xx_total.npy')

zz_bins_5 = np.load(path + '5_keV/10C_sec/40/zz_vac_bins.npy')
zz_total_5 = np.load(path + '5_keV/10C_sec/40/zz_total.npy')

plt.figure(dpi=600, figsize=[4, 3])

plt.plot(xx_total, zz_total_5, 'C0', label=r'пов-ть для растекания')
plt.plot(xx_bins, zz_bins_5, 'C3', label=r'моделирование 5 кэВ')
plt.plot(xx_REF, zz_REF, 'k--', label=r'$\sigma$=250 нм')

plt.xlim(-1000, 1000)
plt.ylim(0, 800)
# plt.ylim(50, 350)

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='upper right')

plt.grid()

plt.savefig('resolution_5.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% 20 keV
xx_bins = np.load(path + '20_keV/10C_sec/30/xx_bins.npy')
xx_total = np.load(path + '20_keV/10C_sec/30/xx_total.npy')

zz_bins_20 = np.load(path + '20_keV/10C_sec/30/zz_vac_bins.npy')
zz_total_20 = np.load(path + '20_keV/10C_sec/30/zz_total.npy')

plt.figure(dpi=600, figsize=[4, 3])

plt.plot(xx_total, zz_total_20, 'C0', label=r'пов-ть для растекания')
plt.plot(xx_bins, zz_bins_20, 'C3', label=r'20 кэВ')
plt.plot(xx_REF, zz_REF, 'k--', label=r'$\sigma$=250 нм')

plt.xlim(-1000, 1000)
plt.ylim(0, 800)

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='upper right')

plt.grid()
plt.savefig('resolution_20.jpg', dpi=600, bbox_inches='tight')
plt.show()
