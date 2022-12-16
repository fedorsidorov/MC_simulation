import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)

xx_REF = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/'


# %% 130C 100s
path = '/Volumes/TOSHIBA EXT/SIM_DEBER_final/130C_100s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 100

zz_bins_sum = np.zeros(len(xx_bins))


for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')


zz_bins_avg = zz_bins_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])

# plt.plot(xx_bins, np.ones(len(xx_bins)) * 500, label='$t = 0$')
plt.plot(xx_bins / 1000, zz_bins_avg, 'C3', label='моделирование')

plt.title(r'130$^\circ$C, 100 c', fontsize=14)
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10)

plt.xlim(-1.5, 1.5)
plt.ylim(0, 800)
plt.grid()

# plt.savefig('130C_100s_14.jpg', dpi=600, bbox_inches='tight')
plt.show()

# %%
np.save('notebooks/DEBER_cooling/130С_100s_REF_xx_bins.npy', xx_bins)
np.save('notebooks/DEBER_cooling/130С_100s_REF_zz_bins_avg.npy', zz_bins_avg)






