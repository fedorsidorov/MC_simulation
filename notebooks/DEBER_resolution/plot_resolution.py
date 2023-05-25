import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)

xx_REF = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/150C_resolution/sigma_5nm/'


# %% 5 keV
xx_bins = np.load(path + '5_keV/10C_sec/40/xx_bins.npy')
xx_total = np.load(path + '5_keV/10C_sec/40/xx_total.npy')

zz_bins_20 = np.load(path + '5_keV/10C_sec/40/zz_vac_bins.npy')
zz_total_20 = np.load(path + '5_keV/10C_sec/40/zz_total.npy')

# plt.figure(dpi=600, figsize=[4, 3])
# plt.figure(dpi=300, figsize=[4, 3])

# plt.plot(xx_REF, np.ones(len(xx_REF)) * 500, 'C3--', label=r'начальная поверхность')
plt.plot(xx_REF / 1000, np.ones(len(xx_REF)) * 500, 'C3--', label=r'начальная поверхность')
# plt.plot(xx_REF, zz_REF, 'k--', label=r'$E=20$ кэВ, $d_{beam}=600$ нм')
plt.plot(xx_REF / 1000, zz_REF, 'k--', label=r'$E=20$ кэВ, $d_{beam}=600$ нм')
# plt.plot(xx_total, zz_total_20, 'C0', label=r'поверхность для растекания')
plt.plot(xx_total / 1000, zz_total_20, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins, zz_bins_20, 'C3', label=r'$E=5$ кэВ, $d_{beam}=10$ нм')
plt.plot(xx_bins / 1000, zz_bins_20, 'C3', label=r'$E=5$ кэВ, $d_{beam}=10$ нм')

# plt.xlim(-1500, 1500)
# plt.xlim(-1.5, 1.5)
# plt.ylim(0, 1000)

plt.xlim(0.4, 0.6)
# plt.ylim(0, 500)

# plt.xlabel(r'$x$, нм')
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='upper right')

plt.grid()
# plt.savefig('resolution_5.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('resolution_5_um_dpi30.jpg', dpi=300, bbox_inches='tight')
plt.show()


# %% 25 keV
xx_bins = np.load(path + '25_keV/10C_sec/45/xx_bins.npy')
xx_total = np.load(path + '25_keV/10C_sec/45/xx_total.npy')

zz_bins_25 = np.load(path + '25_keV/10C_sec/45/zz_vac_bins.npy')
zz_total_25 = np.load(path + '25_keV/10C_sec/45/zz_total.npy')

# plt.figure(dpi=600, figsize=[4, 3])
# plt.figure(dpi=300, figsize=[4, 3])

# plt.plot(xx_REF, np.ones(len(xx_REF)) * 500, 'C3--', label=r'начальная поверхность')
plt.plot(xx_REF / 1000, np.ones(len(xx_REF)) * 500, 'C3--', label=r'начальная поверхность')
# plt.plot(xx_REF, zz_REF, 'k--', label=r'$E=20$ кэВ, $d_{beam}=600$ нм')
plt.plot(xx_REF / 1000, zz_REF, 'k--', label=r'$E=20$ кэВ, $d_{beam}=600$ нм')
# plt.plot(xx_total, zz_total_20, 'C0', label=r'поверхность для растекания')
plt.plot(xx_total / 1000, zz_total_25, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins, zz_bins_20, 'C3', label=r'$E=25$ кэВ, $d_{beam}=10$ нм')
plt.plot(xx_bins / 1000, zz_bins_25, 'C3', label=r'$E=25$ кэВ, $d_{beam}=10$ нм')

# plt.xlim(-1500, 1500)
# plt.xlim(-1.5, 1.5)
# plt.ylim(0, 1000)

plt.xlim(0, 0.4)
plt.ylim(0, 500)

# plt.xlabel(r'$x$, нм')
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10, loc='upper right')

plt.grid()
# plt.savefig('resolution_25.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('resolution_25_um_300dpi.jpg', dpi=300, bbox_inches='tight')
plt.show()
