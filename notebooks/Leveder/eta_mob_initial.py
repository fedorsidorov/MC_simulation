import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import fourier_functions as ff
from functions import MC_functions as mf

mf = importlib.reload(mf)
ff = importlib.reload(ff)

# %% original profile in 2011' paper
xx_um_0 = np.array((0, 0.464, 0.513, 1.5, 1.55, 2)) - 1
xx_um = np.concatenate([xx_um_0 - 4, xx_um_0 - 2, xx_um_0, xx_um_0 + 2, xx_um_0 + 4])

zz_nm_0 = np.array((27.5, 27.5, 55.2, 55.2, 27.5, 27.5))
zz_nm = np.concatenate([zz_nm_0, zz_nm_0, zz_nm_0, zz_nm_0, zz_nm_0])

zz_um = zz_nm * 1e-3

plt.figure(dpi=300)
plt.plot(xx_um, zz_um)
plt.show()

# %%
etas_SI = np.array((1e+2, 3.1e+2, 1e+3, 3.1e+3, 1e+4, 3.1e+4, 1e+5, 3.1e+5,
                    1e+6, 3.1e+6, 1e+7, 3.1e+7, 1e+8, 3.1e+8, 1e+9))
gamma_SI = 34e-3

N = 50
l0_um = 2
l0_m = l0_um * 1e-6

An_array = ff.get_An_array(xx_um * 1e+3, zz_um * 1e+3, l0_um * 1e+3, N) * 1e-9
Bn_array = ff.get_Bn_array(xx_um * 1e+3, zz_um * 1e+3, l0_um * 1e+3, N) * 1e-9

# %% 0
tau_n_array = ff.get_tau_n_easy_array(eta=etas_SI[0], gamma=gamma_SI, h0=An_array[0], l0=l0_m, N=N)

xx_prec_um = np.linspace(-10, 10, 1000)
xx_prec_m = xx_prec_um * 1e-6

zz_0_um = ff.get_h_at_t(xx_prec_m, An_array, Bn_array, tau_n_array, l0_m, t=0)
zz_100_um = ff.get_h_at_t(xx_prec_m, An_array, Bn_array, tau_n_array, l0_m, t=100)

# plt.figure(dpi=600, figsize=[4, 3])
#
# plt.plot(xx_um, zz_um * 1e+3, '-', linewidth='3', label='исходный профиль')
# plt.plot(xx_prec_m * 1e+6, zz_0_um * 1e+9, label='Фурье-профиль')
#
# plt.xlim(-1, 4)
# plt.xlabel(r'$x$, мкм')
# plt.ylabel(r'$z$, нм')
# plt.grid()
# plt.legend()

# plt.savefig('SE_fourier_beg.jpg', dpi=600, bbox_inches='tight')
# plt.show()

# %%
SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/REF_Leveder/vlist_REF_scale_1e-3.txt')
SE = SE[np.where(np.logical_or(np.logical_and(SE[:, 0] == 0, SE[:, 2] > 0.02), SE[:, 1] == -100))]

inds = np.where(SE[:, 1] == -100)[0]
scales = []
profiles = []
now_pos = 0

for ind in inds[1:]:
    now_scale = SE[now_pos, 0]
    now_profile = SE[(now_pos + 1):ind, 1:]
    scales.append(now_scale)
    profiles.append(now_profile)
    now_pos = ind

# %%
ind = 10

plt.figure(dpi=300)
plt.plot(profiles[0][:, 0], profiles[0][:, 1], '.-')
plt.plot(profiles[ind][:, 0], profiles[ind][:, 1], '.')
plt.show()

# %%
# now_eta = etas_SI[0]
# tt = [0.1, 0.5, 1, 2, 3]
# inds = [7, 14, 28, 54, 83]

# now_eta = etas_SI[1]
# tt = [1, 2, 4, 6, 10]
# inds = [10, 18, 36, 54, 90]

# now_eta = etas_SI[2]
# tt = [4, 8, 14, 20, 30]
# inds = [11, 22, 38, 55, 85]

# now_eta = etas_SI[3]
# tt = [10, 25, 40, 55, 90]
# inds = [9, 22, 36, 50, 82]

# now_eta = etas_SI[4]
# tt = [30, 80, 140, 230, 400]
# inds = [9, 22, 39, 65, 110]

# now_eta = etas_SI[5]
# tt = [80, 250, 500, 700, 1200]
# inds = [8, 22, 45, 63, 110]

now_eta = etas_SI[6]
tt = [200, 800, 1300, 2000, 3800]
inds = [8, 22, 37, 56, 110]

# now_eta = etas_SI[7]
# tt = [600, 2000, 4000, 6000, 10000]
# inds = [8, 18, 36, 54, 90]

# now_eta = etas_SI[8]
# tt = [2000, 7000, 12000, 20000, 30000]
# inds = [7, 19, 34, 55, 86]

tau_n_array = ff.get_tau_n_easy_array(eta=now_eta, gamma=gamma_SI, h0=An_array[0], l0=l0_m, N=N)


plt.figure(dpi=600, figsize=[4, 3])

xx_0 = xx_prec_um
zz_0 = ff.get_h_at_t(xx_prec_m, An_array, Bn_array, tau_n_array, l0_m, t=0) * 1e+6

plt.plot(xx_0, zz_0 * 1e+3, 'k', linewidth=1.5, label='$t$ = 0 с')

for i in range(len(tt)):
    zz_t_um = ff.get_h_at_t(xx_prec_m, An_array, Bn_array, tau_n_array, l0_m, t=tt[i]) * 1e+6
    plt.plot(xx_prec_um, zz_t_um * 1e+3, label='$t$ = ' + str(tt[i]) + ' c')
    # plt.plot(profiles[inds[i]][:, 0], profiles[inds[i]][:, 1] * 1e+3, '--', label='s = ' + str(scales[inds[i]]))

for i in range(len(tt)):
    zz_t_um = ff.get_h_at_t(xx_prec_m, An_array, Bn_array, tau_n_array, l0_m, t=tt[i]) * 1e+6
    # plt.plot(xx_prec_um, zz_t_um * 1e+3, label='t = ' + str(tt[i]) + ' c')
    plt.plot(profiles[inds[i]][:, 0], profiles[inds[i]][:, 1] * 1e+3, '--', label='$s$ = ' + str(scales[inds[i]]))

plt.legend(fontsize=10, loc='center right')
# plt.title(r'$\eta$ = ' + str(format(now_eta, '.1e')) + r' Па$\cdot$с')
plt.title(r'$\eta = 10^5$ Па$\cdot$с')
plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')

plt.grid()
plt.xlim(-1, 4)
plt.ylim(20, 60)

plt.savefig('grating_eta_' + str(int(now_eta)) + '.jpg', dpi=600, bbox_inches='tight')
plt.show()
