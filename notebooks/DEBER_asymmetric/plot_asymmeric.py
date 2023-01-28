import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'size': 14}
matplotlib.rc('font', **font)

xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')


# %% sigma = 400 nm, t_exp = 42 s
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_250nm/probs_10C_s/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_250nm/probs_1/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_250nm/probs_3/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_250nm/probs_4/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_250nm/1_3_5_7_our_cooling/'

path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/1_3_5_7_our_cooling/'

# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/1_3_5_7_9_our_cooling/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/1_3_5_7_9_11_our_cooling/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/1_3_5_7_9_11_13_our_cooling/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/pm_500/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/pm_400/'

# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/pm_400/'
# path = '/Volumes/Transcend/SIM_DEBER/366_asymmetric/sigma_200nm/pm_600/'


xx_bins = np.load(path + 'xx_bins.npy')
zz_bins = np.load(path + 'zz_vac_bins.npy')

xx_total = np.load(path + 'xx_total.npy')
zz_total = np.load(path + 'zz_total.npy')

pitch = 3e+3

plt.figure(dpi=600, figsize=[4, 3])

# xx_bins = np.concatenate([xx_bins - pitch, xx_bins, xx_bins + pitch])
# zz_bins = np.concatenate([zz_bins, zz_bins, zz_bins])

xx_bins = np.concatenate([xx_bins - pitch * 2, xx_bins - pitch, xx_bins, xx_bins + pitch, xx_bins + pitch * 2])
zz_bins = np.concatenate([zz_bins, zz_bins, zz_bins, zz_bins, zz_bins])

xx_total = np.concatenate([xx_total - pitch, xx_total, xx_total + pitch])
zz_total = np.concatenate([zz_total, zz_total, zz_total])

# plt.plot(xx_366, zz_366, 'k--')
plt.plot(xx_total / 1e+3, np.ones(len(xx_total)) * 500, 'C3--', label=r'начальная поверхность')
# plt.plot(xx_total, zz_total, 'C0', label=r'пов-ть для растекания')
# plt.plot(xx_bins, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins / 1e+3, zz_bins, 'C3', label=r'моделирование')

plt.legend(fontsize=10, loc='upper right')
# plt.title(r'$\sigma$ = 400 нм, $t_{exp}$ = 42 c', fontsize=14)
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')

# plt.xlim(-4e+3, 4e+3)
plt.xlim(-5, 5)
plt.ylim(0, 800)
plt.grid()

plt.savefig('asymmetric_profile.jpg', dpi=300, bbox_inches='tight')
# plt.savefig('pm_400.jpg', dpi=600, bbox_inches='tight')
plt.show()
