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


# %%
# plt.figure(dpi=600, figsize=[4, 3])
#
# plt.plot([0], [0], 'k--', label=r'$150^\circ$C')
#
# # plt.plot(xx_bins, zz_bins_147_avg, label=r'$147^\circ$C')
# # plt.plot(xx_bins, zz_bins_148_avg, label=r'$148^\circ$C')
# plt.plot(xx_bins, zz_bins_149_avg, label=r'$149^\circ$C')
# # plt.plot(xx_bins, zz_bins_151_avg, label=r'$151^\circ$C')
# # plt.plot(xx_bins, zz_bins_152_avg, label=r'$152^\circ$C')
# plt.plot(xx_bins, zz_bins_153_avg, label=r'$153^\circ$C')
#
# plt.plot(xx_bins, zz_REF - 2, 'k--')
# plt.plot(xx_bins, zz_REF + 2, 'k--')
#
# plt.xlabel(r'$x$, нм')
# plt.ylabel(r'$z$, нм')
# plt.legend(fontsize=10, loc='lower right')
#
# plt.xlim(-1500, 1500)
# # plt.ylim(0, 600)
# plt.ylim(50, 350)
# plt.grid()
#
# # plt.savefig('vary_T_center.jpg', dpi=600, bbox_inches='tight')
# plt.show()

# %%
plt.figure(dpi=300, figsize=[4, 3])

plt.plot(xx_bins / 1000, zz_bins_149_avg, label=r'149 °C')
plt.plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'150 °C')
plt.plot(xx_bins / 1000, zz_REF + 2, 'k--')
plt.plot(xx_bins / 1000, zz_bins_153_avg, label=r'153 °C')
# axs[0, 0].set_xlabel(r'$x$, нм')
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')
# axs[0, 0].set_xlim(-1500, 1500)
plt.xlim(-1.5, 1.5)
plt.ylim(0, 400)
plt.grid()
# plt.legend(fontsize=14, loc='upper center')
plt.legend(fontsize=10, loc='lower right')

plt.savefig('vary_T_4_um_300dpi_SINGLE.jpg', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

axs[0, 0].plot(xx_bins / 1000, zz_bins_149_avg, label=r'149 °C')
axs[0, 0].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'150 °C')
axs[0, 0].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[0, 0].plot(xx_bins / 1000, zz_bins_153_avg, label=r'153 °C')
# axs[0, 0].set_xlabel(r'$x$, нм')
axs[0, 0].set_xlabel(r'$x$, мкм')
axs[0, 0].set_ylabel(r'$z$, нм')
# axs[0, 0].set_xlim(-1500, 1500)
axs[0, 0].set_xlim(-1.5, 1.5)
axs[0, 0].set_ylim(0, 400)
axs[0, 0].grid()
axs[0, 0].legend(fontsize=14, loc='upper center')

axs[1, 0].plot(xx_bins / 1000, zz_bins_149_avg, label=r'149 °C')
axs[1, 0].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'150 °C')
axs[1, 0].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[1, 0].plot(xx_bins / 1000, zz_bins_153_avg, label=r'153 °C')
# axs[1, 0].set_xlabel(r'$x$, нм')
axs[1, 0].set_xlabel(r'$x$, мкм')
axs[1, 0].set_ylabel(r'$z$, нм')
# axs[1, 0].set_xlim(-500, 500)
axs[1, 0].set_xlim(-0.5, 0.5)
axs[1, 0].set_ylim(50, 150)
axs[1, 0].grid()
axs[1, 0].legend(fontsize=14, loc='lower right')

axs[0, 1].plot(xx_bins / 1000, zz_bins_149_avg, label=r'149 °C')
axs[0, 1].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'150 °C')
axs[0, 1].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[0, 1].plot(xx_bins / 1000, zz_bins_153_avg, label=r'153 °C')
# axs[0, 1].set_xlabel(r'$x$, нм')
axs[0, 1].set_xlabel(r'$x$, мкм')
axs[0, 1].set_ylabel(r'$z$, нм')
axs[0, 1].set_xlim(0.5, 1.5)
axs[0, 1].set_ylim(250, 350)
axs[0, 1].grid()
axs[0, 1].legend(fontsize=14, loc='lower right')

axs[1, 1].plot(xx_bins / 1000, zz_bins_149_avg, label=r'149 °C')
axs[1, 1].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'150 °C')
axs[1, 1].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[1, 1].plot(xx_bins / 1000, zz_bins_153_avg, label=r'153 °C')
# axs[1, 1].set_xlabel(r'$x$, нм')
axs[1, 1].set_xlabel(r'$x$, мкм')
axs[1, 1].set_ylabel(r'$z$, нм')
axs[1, 1].set_xlim(0, 1)
axs[1, 1].set_ylim(150, 250)
axs[1, 1].grid()
axs[1, 1].legend(fontsize=14, loc='lower right')

plt.savefig('vary_T_4_um_300dpi.jpg', dpi=300, bbox_inches='tight')
plt.show()



