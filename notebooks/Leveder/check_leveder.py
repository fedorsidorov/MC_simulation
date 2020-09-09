import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import DEBER_functions as deber
from functions import MC_functions as mf

mf = importlib.reload(mf)
deber = importlib.reload(deber)

# %%
profile_0 = np.loadtxt('notebooks/Leveder/2010/0.txt')
profile_100 = np.loadtxt('notebooks/Leveder/2010/100.txt')
profile_200 = np.loadtxt('notebooks/Leveder/2010/200.txt')
profile_500 = np.loadtxt('notebooks/Leveder/2010/500.txt')
profile_1200 = np.loadtxt('notebooks/Leveder/2010/1200.txt')

# eta = 3.23e+5
eta = 3.23e+4
gamma = 34e-3

N = 20
h0 = 43e-9
l0 = 2e-6
T_C = 145

profile_0[0, 0] = 0
profile_0[-1, 0] = l0 * 1e+6

xx = np.linspace(0, l0, 100)
zz = mf.lin_lin_interp(profile_0[:, 0] * 1e-6, profile_0[:, 1] * 1e-9 + h0)(xx)

# xx = profile_0[:, 0] * 1e-6  # um -> m
# zz = profile_0[:, 1] * 1e-9 + h0  # nm -> m
xx[0] = 0
xx[-1] = l0

An_array, Bn_array, tau_n_array = deber.get_An_Bn_tau_arrays(N, l0, h0, eta, gamma, xx, zz)

# %%
zz_0 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 0)
zz_100 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 100)
zz_200 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 200)
zz_500 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 500)
zz_700 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 700)
zz_1000 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 1000)
zz_1200 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 1200)
zz_5000 = deber.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 5000)

# %%
plt.figure(dpi=300)
# plt.plot(xx * 1e+6, zz * 1e+9, 'o', label='0')
# plt.plot(profile_100[:, 0], profile_100[:, 1] + h0*1e+9, 'o', label='100')
# plt.plot(profile_200[:, 0], profile_200[:, 1] + h0*1e+9, 'o', label='200')
# plt.plot(profile_500[:, 0], profile_500[:, 1] + h0*1e+9, 'o', label='500')
# plt.plot(profile_1200[:, 0], profile_1200[:, 1] + h0*1e+9, 'o', label='1200')

plt.plot(xx * 1e+6, zz_0 * 1e+9, '--')
plt.plot(xx * 1e+6, zz_100 * 1e+9, '--')
plt.plot(xx * 1e+6, zz_200 * 1e+9, '--')
plt.plot(xx * 1e+6, zz_500 * 1e+9, '--')
plt.plot(xx * 1e+6, zz_1200 * 1e+9, '--')
plt.plot(xx * 1e+6, zz_5000 * 1e+9, '--')

plt.grid()
# plt.xlim(0, 1)
# plt.ylim(-15, 25)
plt.legend()
plt.show()

# %%
np.save('notebooks/Leveder/2010_sim/yy.npy', xx * 1e+6)
np.save('notebooks/Leveder/2010_sim/0.npy', zz_0 * 1e+9)
np.save('notebooks/Leveder/2010_sim/100.npy', zz_100 * 1e+9)
np.save('notebooks/Leveder/2010_sim/200.npy', zz_200 * 1e+9)
np.save('notebooks/Leveder/2010_sim/500.npy', zz_500 * 1e+9)
np.save('notebooks/Leveder/2010_sim/700.npy', zz_700 * 1e+9)
np.save('notebooks/Leveder/2010_sim/1000.npy', zz_1000 * 1e+9)
np.save('notebooks/Leveder/2010_sim/1200.npy', zz_1200 * 1e+9)
