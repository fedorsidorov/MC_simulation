import numpy as np
import matplotlib.pyplot as plt
import importlib
from functions import boyd_functions as bf

bf = importlib.reload(bf)


# %%
def dose2time(D, I):
    A = 1e-1 * 1.3e-1
    j = I / A
    return D / j


# %%
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

nn = np.arange(1, 50000)
Pn = nn**z_0 * np.exp(-nn / y_0)
Pn /= np.sum(nn**z_0 * np.exp(-nn / y_0))

# plt.figure(dpi=300)
# plt.semilogx(nn, Pn, '-o')
# plt.show()

M1_0 = np.sum(Pn * nn)


# %% get zip lens
zip_len_term_150, _ = bf.get_zip_len_term_trans(150)
# zip_len_term_150 = 150


# %%
tau_total = 400
tau_step = 0.01
tau = np.arange(0, tau_total, tau_step)

solution_150_term = bf.RK4_PCH(zip_len_term_150**-1 * y_0, np.array([1, 1, z_0]), tau)

# %%
M1w_150_term = solution_150_term[:, 0]
yw_150_term = solution_150_term[:, 1]
z_150_term = solution_150_term[:, 2]
y_150_term = yw_150_term * y_0
x_150_term = y_150_term * (z_150_term + 1)

# %%
Mn_150_term = y_150_term * (z_150_term + 1)
Mw_150_term = (z_150_term + 2) / (z_150_term + 1) * Mn_150_term

# %%
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_150_term.npy', Mn_150_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_150_term.npy', Mw_150_term)

# %% plot curves for termination
kin_curve_125 = np.loadtxt('notebooks/Boyd_Schulz_Zimm/kinetic_curves/3.txt')
tt_125 = dose2time(kin_curve_125[:, 0] * 1e-6, 1e-9)
L_norm_125 = kin_curve_125[:, 1]

kin_curve_150 = np.loadtxt('notebooks/Boyd_Schulz_Zimm/kinetic_curves/2.txt')
tt_150 = dose2time(kin_curve_150[:, 0] * 1e-6, 1e-9)
L_norm_150 = kin_curve_150[:, 1]

kin_curve_170 = np.loadtxt('notebooks/Boyd_Schulz_Zimm/kinetic_curves/1.txt')
tt_170 = dose2time(kin_curve_170[:, 0] * 1e-6, 1e-9)
L_norm_170 = kin_curve_170[:, 1]

# plt.figure(dpi=300)

# font_size = 8

# fig, ax = plt.subplots(dpi=300)
fig, ax = plt.subplots(dpi=300, figsize=[4, 3])
# fig.set_size_inches(4, 3)

# plt.plot(tau * 1, M1w_98_term, label='sim T = 98 C°')
# plt.plot(tau * 0.5, M1w_118_term, label='sim T = 118 C°')

plt.plot(tau * 8.5, M1w_125_term, label='simulation T = 125 C°')
plt.plot(tau * 4, M1w_150_term, label='simulation T = 150 C°')
plt.plot(tau * 2.5, M1w_170_term, label='simulation T = 170 C°')

# plt.plot(tau * 8.5, M1w_125_term, label='sim T = 125 C°')
# plt.plot(tau * 4, M1w_150_term, label='sim T = 150 C°')

# plt.plot(dose2time(doses_98_6 * 1e-6, 6e-9), L_norm, 'o', label='exp T = 98 C°, 6 nA')
# plt.plot(dose2time(doses_118_1 * 1e-6, 6e-9), L_norm, 'o', label='exp T = 118 C°, 6 nA')

plt.plot(tt_125, L_norm_125, 'o', label='experiment T = 125 C°')
plt.plot(tt_150, L_norm_150, 'o', label='experiment T = 150 C°')
plt.plot(tt_170, L_norm_170, 'o', label='experiment T = 170 C°')

# plt.plot(dose2time(doses_125_1um * 1e-6, 1e-9), L_norm, 'o', label='exp T = 125 C°, 1 nA, 1000 nm')
# plt.plot(dose2time(doses_150_1um * 1e-6, 1e-9), L_norm, 'o', label='exp T = 150 C°, 1 nA, 1000 nm')

# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)

plt.xlim(0, 10)

plt.legend()
# plt.legend(fontsize=font_size)

plt.title(r'd$_{PMMA}$ = 80 nm, Dose = 16 $\mu$C/cm$^2$')
plt.xlabel('t, s')
plt.ylabel('L$_{norm}$')
# plt.xlabel('t, s', fontsize=font_size)
# plt.ylabel('L$_{norm}$', fontsize=font_size)

plt.xlim(0, 200)
plt.ylim(0, 1)
# plt.grid()
plt.show()

# plt.savefig('for_report.jpg', bbox_inches='tight')

# # %% plot curves for transfer
# plt.figure(dpi=300)

# plt.plot(tau * 1, M1w_98_trans, label='sim T = 98 C°')
# plt.plot(tau * 0.5, M1w_118_trans, label='sim T = 118 C°')
# plt.plot(tau * 0.5, M1w_118_trans, label='sim T = 118 C°')
# plt.plot(tau * 8.5, M1w_125_trans, label='sim T = 125 C°')
# plt.plot(tau * 4, M1w_150_trans, label='sim, T = 150 C°')
# plt.plot(tau * 2.5, M1w_170_trans, label='sim, T = 170 C°')

# plt.plot(dose2time(doses_98_6 * 1e-6, 6e-9), L_norm, 'o', label='exp T = 98 C°, 6 nA')
# plt.plot(dose2time(doses_118_1 * 1e-6, 6e-9), L_norm, 'o', label='exp T = 118 C°, 6 nA')
# plt.plot(tt_125, L_norm_125, 'o', label='exp T = 125 C°')
# plt.plot(tt_150, L_norm_150, 'o', label='exp T = 150 C°')
# plt.plot(tt_170, L_norm_170, 'o', label='exp T = 170 C°')

# plt.xlabel('time, s')
# plt.ylabel('L$_{norm}$')
# plt.xlim(0, 200)
# plt.ylim(0, 1)
# plt.legend()
# plt.grid()
# plt.show()

# %%
tau = np.load('notebooks/Boyd_Schulz_Zimm/arrays/tau.npy')
Mw_125 = np.load('notebooks/Boyd_Schulz_Zimm/arrays/Mw_125.npy')

plt.figure(dpi=300)
# plt.figure(dpi=300, figsize=[4, 3])
# plt.figure(dpi=300, figsize=[6, 4])
# plt.figure(dpi=300, figsize=[8, 6])
plt.semilogy(tau, Mw_130_trans)
plt.semilogy(tau, Mw_130_term, '--')
# plt.semilogy(tau, Mw_125)
# plt.semilogy(tau, Mw_170_term, '--')
# plt.grid()
plt.show()

# size = fig.get_size_inches()
