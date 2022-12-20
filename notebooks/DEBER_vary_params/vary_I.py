import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
xx_bins = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/366_vary_params/'

zz_bins_1p18_sum = np.zeros(len(xx_bins))
zz_bins_1p25_sum = np.zeros(len(xx_bins))

n_tries = 10

for n_try in range(n_tries):
    zz_bins_1p18_sum += np.load(path + 'I_1p18/try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_1p25_sum += np.load(path + 'I_1p25/try_' + str(n_try) + '/zz_vac_bins.npy')

zz_bins_1p18_avg = zz_bins_1p18_sum / n_tries
zz_bins_1p25_avg = zz_bins_1p25_sum / n_tries

# %%
# plt.figure(dpi=600, figsize=[4, 3])
# # plt.figure(dpi=600)
#
# plt.plot([0], [0], 'k--', label=r'4.56 нА')
#
# # plt.plot(xx_bins, zz_bins_1p1_avg, label=r'1.15 нА')
#
# # plt.plot(xx_bins, zz_bins_1p1_avg, label=r'1.18 нА')
# plt.plot(xx_bins, zz_bins_1p1_avg, label=r'4.48 нА')
#
# # plt.plot(xx_bins, zz_bins_1p3_avg, label=r'1.25 нА')
# plt.plot(xx_bins, zz_bins_1p3_avg, label=r'4.75 нА')
#
# plt.plot(xx_bins, zz_REF - 2, 'k--')
# plt.plot(xx_bins, zz_REF + 2, 'k--')
#
# plt.xlim(-1500, 1500)
# plt.ylim(50, 350)
#
# plt.xlabel(r'$x$, нм')
# plt.ylabel(r'$z$, нм')
# plt.legend(fontsize=10, loc='lower right')
#
# plt.grid()
#
# # plt.savefig('vary_I.jpg', dpi=600, bbox_inches='tight')
# plt.show()


# %%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

axs[0, 0].plot(xx_bins / 1000, zz_bins_1p18_avg, label=r'4.48 нА')
axs[0, 0].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'4.56 нА')
axs[0, 0].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[0, 0].plot(xx_bins / 1000, zz_bins_1p25_avg, label=r'4.75 нА')
# axs[0, 0].set_xlabel(r'$x$, нм')
axs[0, 0].set_xlabel(r'$x$, мкм')
axs[0, 0].set_ylabel(r'$z$, нм')
# axs[0, 0].set_xlim(-1500, 1500)
axs[0, 0].set_xlim(-1.5, 1.5)
axs[0, 0].set_ylim(0, 400)
axs[0, 0].grid()
axs[0, 0].legend(fontsize=14, loc='upper center')

axs[1, 0].plot(xx_bins / 1000, zz_bins_1p18_avg, label=r'4.48 нА')
axs[1, 0].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'4.56 нА')
axs[1, 0].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[1, 0].plot(xx_bins / 1000, zz_bins_1p25_avg, label=r'4.75 нА')
# axs[1, 0].set_xlabel(r'$x$, нм')
axs[1, 0].set_xlabel(r'$x$, мкм')
axs[1, 0].set_ylabel(r'$z$, нм')
axs[1, 0].set_xlim(-0.5, 0.5)
axs[1, 0].set_ylim(50, 150)
axs[1, 0].grid()
axs[1, 0].legend(fontsize=14, loc='lower right')

axs[0, 1].plot(xx_bins / 1000, zz_bins_1p18_avg, label=r'4.48 нА')
axs[0, 1].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'4.56 нА')
axs[0, 1].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[0, 1].plot(xx_bins / 1000, zz_bins_1p25_avg, label=r'4.75 нА')
# axs[0, 1].set_xlabel(r'$x$, нм')
axs[0, 1].set_xlabel(r'$x$, мкм')
axs[0, 1].set_ylabel(r'$z$, нм')
axs[0, 1].set_xlim(0.5, 1.5)
axs[0, 1].set_ylim(250, 350)
axs[0, 1].grid()
axs[0, 1].legend(fontsize=14, loc='lower right')

axs[1, 1].plot(xx_bins / 1000, zz_bins_1p18_avg, label=r'4.48 нА')
axs[1, 1].plot(xx_bins / 1000, zz_REF - 2, 'k--', label=r'4.56 нА')
axs[1, 1].plot(xx_bins / 1000, zz_REF + 2, 'k--')
axs[1, 1].plot(xx_bins / 1000, zz_bins_1p25_avg, label=r'4.75 нА')
# axs[1, 1].set_xlabel(r'$x$, нм')
axs[1, 1].set_xlabel(r'$x$, мкм')
axs[1, 1].set_ylabel(r'$z$, нм')
axs[1, 1].set_xlim(0, 1)
axs[1, 1].set_ylim(150, 250)
axs[1, 1].grid()
axs[1, 1].legend(fontsize=14, loc='lower right')

plt.savefig('vary_I_4_um_300dpi.jpg', dpi=300, bbox_inches='tight')
plt.show()