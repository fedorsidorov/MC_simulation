import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import fourier_functions as ff
from functions import MC_functions as mf

mf = importlib.reload(mf)
rf = importlib.reload(ff)

# %%
profile_0 = np.loadtxt('notebooks/Leveder/2011/profile_0.txt')
profile_1 = np.loadtxt('notebooks/Leveder/2011/profile_1.txt')
profile_2 = np.loadtxt('notebooks/Leveder/2011/profile_2.txt')
profile_3 = np.loadtxt('notebooks/Leveder/2011/profile_3.txt')
profile_4 = np.loadtxt('notebooks/Leveder/2011/profile_4.txt')
profile_5 = np.loadtxt('notebooks/Leveder/2011/profile_5.txt')

N = 100
h0 = 27e-9
l0 = 2e-6
T_C = 145

# eta = 3.23e+5
# eta = 3.23e+4
eta = 3e+4
gamma = 34e-3

tau_1 = 369

# eta_test = tau_1 / 3 * gamma * h0**3 / (l0 / 2 / np.pi)**4
# eta = eta_test

# %%
profile_0[0, 0] = 0
profile_0[-1, 0] = l0 * 1e+6

xx = np.linspace(0, l0, 100)
zz = mf.lin_lin_interp(profile_0[:, 0] * 1e-6, profile_0[:, 1] * 1e-9)(xx)

# xx = profile_0[:, 0] * 1e-6  # um -> m
# zz = profile_0[:, 1] * 1e-9 + h0  # nm -> m
# xx[0] = 0
# xx[-1] = l0
xx -= l0/2
xx[0] = -l0 / 2
xx[-1] = l0 / 2

plt.figure(dpi=300)
plt.plot(xx * 1e+6, zz * 1e+9)
plt.show()

# %%
An_array = ff.get_An_array(xx * 1e+9, zz * 1e+9, l0 * 1e+9, N) * 1e-9
Bn_array = ff.get_Bn_array(xx * 1e+9, zz * 1e+9, l0 * 1e+9, N) * 1e-9
tau_n_array = ff.get_tau_n_easy_array(eta, gamma, h0, l0, N)

# %%
zz_0 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 0)
zz_30 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 30)
zz_130 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 130)
zz_230 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 230)
zz_330 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 330)
zz_480 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 480)
zz_630 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 630)
zz_880 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 880)
zz_1180 = ff.get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, 1180)

# %%
plt.figure(dpi=300)
plt.plot(xx * 1e+6, zz * 1e+9, 'o', label='0')
plt.plot(profile_0[:, 0] - 1, profile_0[:, 1], 'o', label='0')
plt.plot(profile_1[:, 0] - 1, profile_1[:, 1], 'o', label='1')
plt.plot(profile_2[:, 0] - 1, profile_2[:, 1], 'o', label='2')
plt.plot(profile_3[:, 0] - 1, profile_3[:, 1], 'o', label='3')
plt.plot(profile_4[:, 0] - 1, profile_4[:, 1], 'o', label='4')
plt.plot(profile_5[:, 0] - 1, profile_5[:, 1], 'o', label='5')

plt.plot(xx * 1e+6, zz_0*1e+9, '--')
plt.plot(xx * 1e+6, zz_30*1e+9, '--')
plt.plot(xx * 1e+6, zz_130*1e+9, '--')
plt.plot(xx * 1e+6, zz_230*1e+9, '--')
plt.plot(xx * 1e+6, zz_330*1e+9, '--')
plt.plot(xx * 1e+6, zz_480*1e+9, '--')
plt.plot(xx * 1e+6, zz_630*1e+9, '--')
plt.plot(xx * 1e+6, zz_880*1e+9, '--')
plt.plot(xx * 1e+6, zz_1180*1e+9, '--')

plt.grid()
# plt.xlim(0, 1)
# plt.ylim(-15, 25)
plt.legend()
plt.show()

# %%
now_zz = zz_1180
(now_zz.max() - now_zz.min()) * 1e+9

