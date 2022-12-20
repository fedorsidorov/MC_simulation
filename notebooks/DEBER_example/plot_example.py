import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functions import MC_functions as mcf

mcf = importlib.reload(mcf)

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
path = '/Volumes/Transcend/SIM_DEBER/150C_100s_example/2_nA/1C_sec/'

xx_bins = np.load(path + 'xx_bins.npy')
zz_bins = np.load(path + 'zz_vac_bins.npy')

xx_total = np.load(path + 'xx_total.npy')
zz_total = np.load(path + 'zz_total.npy')

# ind = 1
# ind = 5
# ind = 22
# ind = 30
# ind = 94
ind = 99

xx_bins_now = np.load(path + 'xx_bins_' + str(ind) + '.npy')
zz_bins_now = np.load(path + 'zz_vac_bins_' + str(ind) + '.npy')
xx_total_now = np.load(path + 'xx_total_' + str(ind) + '.npy')
zz_total_now = np.load(path + 'zz_total_' + str(ind) + '.npy')

xx_beam_now = np.load(path + 'now_x0_array_' + str(ind) + '.npy')
zz_beam_now = mcf.lin_lin_interp(xx_bins, zz_bins_now)(xx_beam_now)


# %
plt.figure(dpi=300, figsize=[4, 3])

plt.plot(xx_total / 1000, np.ones(len(xx_total)) * 500, 'C3--', label=r'начальная поверхность')
plt.plot(xx_total_now / 1000, zz_total_now, 'C0', label=r'поверхность для растекания')
plt.plot(xx_bins_now / 1000, zz_bins_now, 'C3', label=r'поверхность ПММА')
plt.plot(xx_beam_now / 1000, zz_beam_now, 'k.', label=r'электронный луч')

# plt.plot(xx_total / 1000, np.ones(len(xx_total)) * 500, 'C3--', label=r'начальная поверхность')
# plt.plot(xx_total / 1000, zz_total, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins, 'C3', label=r'поверхность ПММА')

plt.legend(fontsize=10, loc='upper right')

plt.title(r'$t$ = ' + str(ind) + ' с', fontsize=14)
# plt.title(r'$t$ = 170 с', fontsize=14)
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')

plt.xlim(-1.5, 1.5)
plt.ylim(0, 1000)
plt.grid()

plt.savefig('example_' + str(ind) + '.jpg', dpi=300, bbox_inches='tight')
# plt.savefig('example_170.jpg', dpi=300, bbox_inches='tight')
plt.show()




