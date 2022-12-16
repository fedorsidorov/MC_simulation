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

n_tries = 100

zz_bins_sum = np.zeros(len(xx_bins))

plt.figure(dpi=600)

for n_try in range(n_tries):
    now_zz_bins = np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_bins_sum += now_zz_bins
    # plt.plot(xx_bins, now_zz_bins)


zz_bins_avg = zz_bins_sum / n_tries

# plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.plot(xx_366, zz_366 + 75, 'k--')
# plt.plot(xx_366, zz_366 + 100, 'k--')
# plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
# plt.grid()
# plt.show()


# %%
quad_error_arr = np.zeros(len(xx_bins))

for n_try in range(n_tries):
    now_zz_bins = np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')
    quad_error_arr += (now_zz_bins - zz_bins_avg)**2

mean_quad_error_arr = np.sqrt(quad_error_arr / n_tries)

plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.plot(xx_bins, zz_bins_avg + mean_quad_error_arr, 'k--')
# plt.plot(xx_bins, zz_bins_avg - mean_quad_error_arr, 'k--')
plt.plot(xx_bins, mean_quad_error_arr)
# plt.plot(xx_366, zz_366 + 75, 'k--')
# plt.plot(xx_366, zz_366 + 100, 'k--')
plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
plt.grid()
plt.show()


# %% 150C 200s
path = '/Volumes/Transcend/SIM_DEBER/150C_200s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 73

zz_bins_sum = np.zeros(len(xx_bins))


for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')


zz_bins_avg = zz_bins_sum / n_tries

# plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
# plt.grid()
# plt.show()


# %
quad_error_arr = np.zeros(len(xx_bins))

for n_try in range(n_tries):
    now_zz_bins = np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')

    quad_error_arr += (now_zz_bins - zz_bins_avg)**2

mean_quad_error_arr = np.sqrt(quad_error_arr / n_tries)

plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.plot(xx_bins, zz_bins_avg + mean_quad_error_arr, 'k--')
# plt.plot(xx_bins, zz_bins_avg - mean_quad_error_arr, 'k--')
plt.plot(xx_bins, mean_quad_error_arr)
plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
plt.grid()
plt.show()


# %% 130C 100s
path = '/Volumes/Transcend/SIM_DEBER/130C_100s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 100

zz_bins_sum = np.zeros(len(xx_bins))


for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')


zz_bins_avg = zz_bins_sum / n_tries

# plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
# plt.grid()
# plt.show()


# %
quad_error_arr = np.zeros(len(xx_bins))

for n_try in range(n_tries):
    now_zz_bins = np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')

    quad_error_arr += (now_zz_bins - zz_bins_avg)**2

mean_quad_error_arr = np.sqrt(quad_error_arr / n_tries)

plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx_bins, zz_bins_avg)
# plt.plot(xx_bins, zz_bins_avg + mean_quad_error_arr, 'k--')
# plt.plot(xx_bins, zz_bins_avg - mean_quad_error_arr, 'k--')
plt.plot(xx_bins, mean_quad_error_arr)
plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
plt.grid()
plt.show()

# np.save('notebooks/DEBER_simulation/sigma_150C_200s.npy', mean_quad_error_arr)


# %% 130C 200s
path = '/Volumes/Transcend/SIM_DEBER/130C_200s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 61

zz_bins_sum = np.zeros(len(xx_bins))


for n_try in range(n_tries):
    zz_bins_sum += np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')


zz_bins_avg = zz_bins_sum / n_tries

# plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
# plt.grid()
# plt.show()


# %
quad_error_arr = np.zeros(len(xx_bins))

for n_try in range(n_tries):
    now_zz_bins = np.load(path + 'try_' + str(n_try) + '/zz_vac_bins.npy')

    quad_error_arr += (now_zz_bins - zz_bins_avg)**2

mean_quad_error_arr = np.sqrt(quad_error_arr / n_tries)

plt.figure(dpi=600, figsize=[4, 3])
# plt.plot(xx_bins, zz_bins_avg)
# plt.plot(xx_bins, zz_bins_avg + mean_quad_error_arr, 'k--')
# plt.plot(xx_bins, zz_bins_avg - mean_quad_error_arr, 'k--')
plt.plot(xx_bins, mean_quad_error_arr)
plt.xlim(-1500, 1500)
# plt.ylim(0, 600)
plt.grid()
plt.show()

# np.save('notebooks/DEBER_simulation/sigma_150C_200s.npy', mean_quad_error_arr)


# %%
sigma_1 = np.load('notebooks/DEBER_simulation/sigma_150C_100s.npy')
sigma_2 = np.load('notebooks/DEBER_simulation/sigma_150C_200s.npy')
sigma_3 = np.load('notebooks/DEBER_simulation/sigma_130C_100s.npy')
sigma_4 = np.load('notebooks/DEBER_simulation/sigma_130C_200s.npy')

# plt.figure(dpi=600, figsize=[4, 3])
plt.figure(dpi=300, figsize=[4, 3])

plt.plot(xx_bins / 1000, sigma_1, label=r'150$^\circ$C, 100 c')
plt.plot(xx_bins / 1000, sigma_2, label=r'150$^\circ$C, 200 c')
plt.plot(xx_bins / 1000, sigma_3, label=r'130$^\circ$C, 100 c')
plt.plot(xx_bins / 1000, sigma_4, label=r'130$^\circ$C, 200 c')

# plt.xlabel(r'$x$, нм')
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$\sigma$, нм')
plt.legend(fontsize=10)

# plt.xlim(-1500, 1500)
plt.xlim(-1.5, 1.5)
plt.ylim(0, 14)
plt.grid()

plt.savefig('DEBER_sigmas_um_300dpi.jpg', dpi=300, bbox_inches='tight')
plt.show()



