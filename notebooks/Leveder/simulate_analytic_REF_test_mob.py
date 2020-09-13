import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import reflow_functions as rf
from functions import MC_functions as mf

mf = importlib.reload(mf)
rf = importlib.reload(rf)

# %% original profile in um
yy_um = np.load('notebooks/SE/REF/yy_pre.npy')
zz_um = np.load('notebooks/SE/REF/zz_pre.npy')

yy_nm = yy_um * 1e+3
zz_nm = zz_um * 1e+3

plt.figure(dpi=300)
plt.plot(yy_nm, zz_nm)
plt.show()

# %%
etas_SI = np.logspace(3, 9, 10)
gamma_SI = 34e-3

N = 20
l0_um = 2
l0_nm = l0_um * 1e+3
l0_m = l0_um * 1e-6

An_array_nm = rf.get_An_array(xx=yy_nm, zz=zz_nm, l0=l0_nm, N=N)
Bn_array_nm = rf.get_Bn_array(xx=yy_nm, zz=zz_nm, l0=l0_nm, N=N)
An_array_m = An_array_nm * 1e-9
Bn_array_m = Bn_array_nm * 1e-9

# %%
tau_n_array_s = rf.get_tau_n_easy_array(eta=etas_SI[3], gamma=gamma_SI, h0=An_array_m[0], l0=l0_m, N=N)

yy_prec_um = np.linspace(-10, 10, 1000)

zz_0_um = rf.get_h_at_t(yy_prec_um, An_array_m, Bn_array_m, tau_n_array_s, l0_m, t=0) * 1e+6
zz_100_um = rf.get_h_at_t(yy_prec_um, An_array_m, Bn_array_m, tau_n_array_s, l0_m, t=100) * 1e+6

# %
plt.figure(dpi=300)
plt.plot(yy_um, zz_um, 'o-', label='original')
plt.plot(yy_prec_um, zz_0_um, label='analytic 0 s')
plt.plot(yy_prec_um, zz_100_um, '.-', label='analytic 100 s')

plt.grid()
plt.legend()
plt.show()

# %%
SE = np.loadtxt('notebooks/SE/vlist.txt')

times = []
profiles = []
beg = -1

for i, line in enumerate(SE):
    if line[1] == line[2] == -100:
        now_time = line[0]
        times.append(now_time)
        profile = SE[beg+1:i, 1:]
        profile = profile[np.where(np.abs(profile[:, 0]) < 5)]
        profile = profile[np.where(profile[:, 1] > 0.03)]
        profile[:, 1] -= (np.max(profile[:, 1]) + np.min(profile[:, 1]))/2 - 0.05
        profiles.append(profile)
        beg = i

# %%
ind = 20

plt.figure(dpi=300)
plt.plot(profiles[0][:, 0], profiles[0][:, 1], '.')
plt.plot(profiles[ind][:, 0], profiles[ind][:, 1], '.')
plt.show()

# %%
eta_ind = 3
tau_n_array_s = rf.get_tau_n_easy_array(eta=etas_SI[eta_ind], gamma=gamma_SI, h0=An_array_m[0], l0=l0_m, N=N)

tt = [100, 200, 400, 1000, 2000]
inds = [2, 4, 8, 20, 40]

plt.figure(dpi=300)

for i in range(len(tt)):
    zz_t_um = rf.get_h_at_t(yy_prec_um, An_array_m, Bn_array_m, tau_n_array_s, l0_m, t=tt[i]) * 1e+6
    plt.plot(yy_prec_um, zz_t_um, label='Analytic, t = ' + str(int(tt[i])) + ' s')
    plt.plot(profiles[inds[i]][:, 0], profiles[inds[i]][:, 1], 'o',
             label='SE, scale = ' + str(int(times[inds[i]])) + ' s')

plt.xlim(-1, 1)
plt.ylim(0.03, 0.07)
plt.title(r'$\eta$ = ' + str(int(etas_SI[eta_ind])) + ' Pa s')
plt.xlabel('x, um')
plt.ylabel('z, um')
plt.legend()
plt.grid()
plt.show()

