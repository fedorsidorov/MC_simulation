import numpy as np
from scipy import interpolate
from mapping import mapping_3um_500nm as mm


def lin_log_interp(xp, yp, kind='linear', axis=-1):
    log_yp = np.log10(yp)
    interp = interpolate.interp1d(xp, log_yp, kind=kind, axis=axis)
    def func(x): return np.power(10.0, interp(x))
    return func


tau = np.load('tau.npy')
Mn_150 = np.load('Mn_150.npy') * 100

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

x_min, x_max = -1500, 1500/2
z_min, z_max = 0, 500

x_step, z_step = 100, 5

xx_bins = np.arange(x_min, x_max + 1, 100)
zz_bins = np.arange(z_min, z_max + 1, 5)

xx_centers = (xx_bins[:-1] + xx_bins[1:]) / 2
zz_centers = (zz_bins[:-1] + zz_bins[1:]) / 2

bin_volume = x_step * mm.ly * z_step
V_mon_nm3 = 0.1397
bin_n_monomers = bin_volume / V_mon_nm3

tau_matrix = np.zeros((len(xx_centers), len(zz_centers)))
Mn_matrix = np.ones((len(xx_centers), len(zz_centers))) * Mn_150[0]

exposure_time = 100
time_step = 1
now_time = 0


while now_time < exposure_time:

    now_scission_matrix = np.load('i-th_scission_matrix_path')

    for i in range(len(xx_centers)):
        for j in range(len(zz_centers)):
            now_k_s = now_scission_matrix[i, j] / time_step / bin_n_monomers
            tau_matrix[i, j] += y_0 * now_k_s * time_step
            Mn_matrix[i, j] = lin_log_interp(tau, Mn_150)(tau_matrix[i, j])

    now_time += 1

np.save('Mn_matrix_final.npy', Mn_matrix)
