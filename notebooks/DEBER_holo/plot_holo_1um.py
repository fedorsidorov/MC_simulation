import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'size': 14}
matplotlib.rc('font', **font)


pitch = 1e+3


def func_cos(x_arr, h, A):
    return h - A * np.cos(2 * np.pi / pitch * x_arr)


# %%
# T_step, beam_sigma, t_exp = 1, 200, 15
# T_step, beam_sigma, t_exp = 1, 200, 20
# T_step, beam_sigma, t_exp = 1, 250, 20
# T_step, beam_sigma, t_exp = 1, 300, 15
# T_step, beam_sigma, t_exp = 5, 100, 30
# T_step, beam_sigma, t_exp = 5, 150, 25
# T_step, beam_sigma, t_exp = 5, 200, 25
# T_step, beam_sigma, t_exp = 5, 250, 25
T_step, beam_sigma, t_exp = 5, 300, 25

path = '/Volumes/Transcend/SIM_DEBER/150C_holo_1/' + str(T_step) + 'C_sec/sigma_' + str(beam_sigma) + '/' + str(t_exp) + '/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=300, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2, xx_bins_0 + pitch * 3])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot((xx_total - 1000) / 1000, np.ones(len(xx_total)) * 500, 'C3--')
plt.plot((xx_total - 1000) / 1000, zz_total, 'C0', label=r'поверхность для растекания')
plt.plot((xx_bins - 1000) / 1000, zz_bins, 'C3', label=r'поверхность ПММА')
plt.plot((xx_bins - 1000) / 1000, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10, loc='upper right')
plt.title(r'$\sigma$ = ' + str(beam_sigma) + ' нм, $t_{exp}$ = ' + str(t_exp) + ' c', fontsize=14)
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')

plt.xlim(-2, 2)
plt.ylim(0, 800)
plt.grid()

plt.savefig('holo_' + str(T_step) + 'C_s' + str(beam_sigma) + '_' + str(t_exp) + 's_um_300dpi.jpg', dpi=300, bbox_inches='tight')
# plt.show()
