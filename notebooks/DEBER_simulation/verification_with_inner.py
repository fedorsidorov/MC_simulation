import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')

path = '/Volumes/Transcend/SIM_DEBER/150C_100s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')
xx_total = np.load(path + 'try_0/xx_total.npy')

n_tries = 100

zz_bins_sum = np.zeros(len(xx_bins))
zz_total_sum = np.zeros(len(xx_total))

for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_total_sum += np.load(path + 'try_' + str(n_try) + '/zz_total.npy')

zz_bins_avg = zz_bins_sum / n_tries
zz_total_avg = zz_total_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])

plt.plot(xx_366, zz_366 + 75, 'k--', label='эксперимент')
plt.plot(xx_366, zz_366 + 100, 'k--')
# plt.plot(xx_bins, np.ones(len(xx_bins)) * 500, label='$t = 0$')
plt.plot(xx_bins, zz_bins_avg, 'C3', label='моделирование')
# plt.plot(xx_total, zz_total_avg, 'C0--', label='поверхность для растекания')

plt.title(r'150$^\circ$C, 100 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$y$, нм')
plt.legend(fontsize=10)

plt.xlim(-1500, 1500)
plt.ylim(0, 700)
plt.grid()

plt.savefig('150C_100s_14.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% 150C 200s
xx_356_A = np.load('notebooks/DEBER_simulation/exp_profiles/356/xx_356_C_slice_1.npy')
zz_356_A = np.load('notebooks/DEBER_simulation/exp_profiles/356/zz_356_C_slice_1.npy')

xx_356_B = np.load('notebooks/DEBER_simulation/exp_profiles/356/xx_356_C_slice_3.npy')
zz_356_B = np.load('notebooks/DEBER_simulation/exp_profiles/356/zz_356_C_slice_3.npy')

path = '/Volumes/Transcend/SIM_DEBER/150C_200s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 73

zz_bins_sum = np.zeros(len(xx_bins))


for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')


zz_bins_avg = zz_bins_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])

plt.plot(xx_356_A, zz_356_A, 'k--', label='эксперимент')
plt.plot(xx_356_B, zz_356_B, 'k--')
plt.plot(xx_bins, zz_bins_avg, 'C3', label='моделирование')

plt.title(r'150$^\circ$C, 200 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$y$, нм')
plt.legend(fontsize=10)

plt.xlim(-1500, 1500)
plt.ylim(0, 700)
plt.grid()

plt.savefig('150C_200s_14.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% 130C 100s
xx_360 = np.load('notebooks/DEBER_simulation/exp_profiles/360/xx_360.npy')
zz_360 = np.load('notebooks/DEBER_simulation/exp_profiles/360/zz_360.npy')

path = '/Volumes/Transcend/SIM_DEBER/130C_100s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')
xx_total = np.load(path + 'try_0/xx_total.npy')

n_tries = 100

zz_bins_sum = np.zeros(len(xx_bins))
zz_total_sum = np.zeros(len(xx_total))

for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_total_sum += np.load(path + 'try_' + str(n_try) + '/zz_total.npy')

zz_bins_avg = zz_bins_sum / n_tries
zz_total_avg = zz_total_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])

# plt.plot(xx_366, np.ones(len(xx_366)) * 500, 'C3--', label='начальная поверхность')
plt.plot(xx_366, np.ones(len(xx_366)) * 500, 'C3--')
plt.plot([-1], [-1], 'k--', label='эксперимент')
plt.plot([-1], [-1], 'C3', label='моделирование')
plt.plot(xx_total, zz_total_avg, 'C0', label='поверхность для растекания')
plt.plot(xx_bins, zz_bins_avg, 'C3')
plt.plot(xx_360, zz_360, 'k--')

plt.title(r'130$^\circ$C, 100 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')
plt.legend(fontsize=10)

plt.xlim(-1500, 1500)
plt.ylim(0, 800)
# plt.ylim(0, 700)
plt.grid()

plt.savefig('130C_100s_14_inner_dPMMA.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% 130C 200s
xx_1 = np.load('notebooks/DEBER_simulation/exp_profiles/357/xx_357_lower_slice_3.npy')
zz_1 = np.load('notebooks/DEBER_simulation/exp_profiles/357/zz_357_lower_slice_3.npy')

xx_2 = np.load('notebooks/DEBER_simulation/exp_profiles/357/xx_357_lower_slice_4.npy')
zz_2 = np.load('notebooks/DEBER_simulation/exp_profiles/357/zz_357_lower_slice_4.npy')

path = '/Volumes/Transcend/SIM_DEBER/130C_200s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 61

zz_bins_sum = np.zeros(len(xx_bins))


for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')


zz_bins_avg = zz_bins_sum / n_tries

plt.figure(dpi=600, figsize=[4, 3])


plt.plot(xx_1, zz_1, 'k--', label='эксперимент')
plt.plot(xx_2, zz_2, 'k--')
# plt.plot(xx_bins, np.ones(len(xx_bins)) * 500, label='$t = 0$')
plt.plot(xx_bins, zz_bins_avg, 'C3', label='моделирование')

plt.title(r'130$^\circ$C, 100 c', fontsize=14)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$y$, нм')
plt.legend(fontsize=10)

plt.xlim(-1500, 1500)
plt.ylim(0, 700)
plt.grid()

plt.savefig('130C_200s_14.jpg', dpi=600, bbox_inches='tight')
plt.show()





