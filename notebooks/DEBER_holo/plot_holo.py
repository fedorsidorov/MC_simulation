import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'size': 14}
matplotlib.rc('font', **font)


pitch = 2e+3


def func_cos(x_arr, h, A):
    return h - A * np.cos(2 * np.pi / pitch * x_arr)


# %% sigma = 400 nm, t_exp = 42 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_400/42/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 400 нм, $t_{exp}$ = 42 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_400_42.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 450 nm, t_exp = 40 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_450/40/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 450 нм, $t_{exp}$ = 40 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_450_40.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 450 nm, t_exp = 48 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_450/48/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 450 нм, $t_{exp}$ = 48 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_450_48.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 500 nm, t_exp = 48 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_500/48/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 500 нм, $t_{exp}$ = 48 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_500_48.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 600 nm, t_exp = 40 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_600/40/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 600 нм, $t_{exp}$ = 40 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_600_40.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 600 nm, t_exp = 60 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_600/60/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 600 нм, $t_{exp}$ = 60 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_600_60.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 600 nm, t_exp = 62 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/1C_sec/sigma_600/62/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 600 нм, $t_{exp}$ = 62 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_600_62.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% sigma = 250 nm, t_exp = 100 s
path = '/Volumes/Transcend/SIM_DEBER/150C_holo/10C_sec/sigma_250/100/'

xx_bins_0 = np.load(path + 'xx_bins.npy')
zz_bins_0 = np.load(path + 'zz_vac_bins.npy')

xx_total_0 = np.load(path + 'xx_total.npy')
zz_total_0 = np.load(path + 'zz_total.npy')


plt.figure(dpi=600, figsize=[4, 3])

xx_bins = np.concatenate([xx_bins_0 - pitch, xx_bins_0, xx_bins_0 + pitch, xx_bins_0 + pitch * 2])
zz_bins = np.concatenate([zz_bins_0, zz_bins_0, zz_bins_0, zz_bins_0])

xx_total = np.concatenate([xx_total_0, xx_total_0 + pitch * 3])
zz_total = np.concatenate([zz_total_0, zz_total_0])

popt, _ = curve_fit(func_cos, xx_bins, zz_bins)

plt.plot(xx_total - 500, zz_total, 'C0', label=r'поверхность ПММА')
plt.plot(xx_bins - 500, zz_bins, 'C3', label=r'моделирование')
plt.plot(xx_bins - 500, func_cos(xx_bins, *popt), 'k--', label=r'$h_0 + A \cdot sin(2 \pi x / \lambda)$')

plt.legend(fontsize=10)
plt.title(r'$\sigma$ = 250 нм, $t_{exp}$ = 100 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

plt.xlim(-3e+3, 3e+3)
plt.ylim(0, 700)
plt.grid()

plt.savefig('holo_250_100_10C_sec.jpg', dpi=600, bbox_inches='tight')
plt.show()








