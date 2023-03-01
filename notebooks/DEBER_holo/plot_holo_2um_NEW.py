import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'size': 14}
matplotlib.rc('font', **font)

pitch = 2e+3


def func_cos(x_arr, h, A):
    return h - A * np.cos(2 * np.pi / pitch * x_arr)


# %%
# T_step, beam_sigma, t_exp = 1, 400, 42
# T_step, beam_sigma, t_exp = 1, 450, 40
# T_step, beam_sigma, t_exp = 1, 450, 48  # 1)
# T_step, beam_sigma, t_exp = 1, 500, 48
# T_step, beam_sigma, t_exp = 1, 600, 48
# T_step, beam_sigma, t_exp = 1, 600, 40  # 2)
# T_step, beam_sigma, t_exp = 1, 600, 62
# T_step, beam_sigma, t_exp = 10, 250, 100  # 3)
T_step, beam_sigma, t_exp = 1, 600, 60  # 4)

path = '/Volumes/Transcend/SIM_DEBER/150C_holo_2/' + str(T_step) + 'C_sec/sigma_' + str(beam_sigma) + '/' + str(t_exp) + '/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')

plt.figure(dpi=300, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

zz_sin = func_cos(xx_bins, *popt)
mean_quad_error_arr = np.abs(zz_bins - zz_sin)
print(np.max(mean_quad_error_arr))

# plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.plot(xx_bins, zz_bins_avg + mean_quad_error_arr, 'k--')
# plt.plot(xx_bins, zz_bins_avg - mean_quad_error_arr, 'k--')
# plt.plot(xx_bins, mean_quad_error_arr)
# plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
# plt.grid()
# plt.show()

plt.plot((xx_total - 2000) / 1000, np.ones(len(xx_total)) * 500, 'C3--')
plt.plot((xx_total - 2000) / 1000, zz_total, 'C0', label=r'поверхность для растекания')
plt.plot((xx_bins - 2000) / 1000, zz_bins, 'C3', label=r'поверхность ПММА')
plt.plot((xx_bins - 2000) / 1000, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10, loc='upper right')
plt.title(r'$\sigma$ = ' + str(beam_sigma) + ' нм, $t_{exp}$ = ' + str(t_exp) + ' c', fontsize=14)
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3, 3)
plt.ylim(0, 800)
plt.grid()

# plt.savefig('holo_' + str(T_step) + 'C_s' + str(beam_sigma) + '_' + str(t_exp) + 's_um_300dpi.jpg', dpi=300, bbox_inches='tight')
plt.show()
