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
zip_len_term_140, _ = bf.get_zip_len_term_trans(140)
zip_len_term_145, _ = bf.get_zip_len_term_trans(145)
zip_len_term_150, _ = bf.get_zip_len_term_trans(150)
zip_len_term_155, _ = bf.get_zip_len_term_trans(155)
zip_len_term_160, _ = bf.get_zip_len_term_trans(160)

# %% 2-point curves
# L_norm = np.array([1, 0.5, 0])
#
# doses_118_1 = np.array([0, 3.8, 30])
# doses_98_6 = np.array([0, 22, 180])
# doses_125_6 = np.array([0, 5.2, 46])
# doses_125_0p15 = np.array([0, 0.85, 8])
#
# doses_125_150nm = np.array([0, 1.3, 10.2])
# doses_125_1um = np.array([0, 20, 170])
# doses_150_1um = np.array([0, 3.6, 30])
# doses_170_2um = np.array([0, 4.0, 33.6])

# %%
tau_total = 400
tau_step = 0.01
tau = np.arange(0, tau_total, tau_step)

solution_140_term = bf.RK4_PCH(zip_len_term_140**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_145_term = bf.RK4_PCH(zip_len_term_145**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_150_term = bf.RK4_PCH(zip_len_term_150**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_155_term = bf.RK4_PCH(zip_len_term_155**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_160_term = bf.RK4_PCH(zip_len_term_160**-1 * y_0, np.array([1, 1, z_0]), tau)

# %%
M1w_140_term = solution_140_term[:, 0]
M1w_145_term = solution_145_term[:, 0]
M1w_150_term = solution_150_term[:, 0]
M1w_155_term = solution_155_term[:, 0]
M1w_160_term = solution_160_term[:, 0]

yw_140_term = solution_140_term[:, 1]
yw_145_term = solution_145_term[:, 1]
yw_150_term = solution_150_term[:, 1]
yw_155_term = solution_155_term[:, 1]
yw_160_term = solution_160_term[:, 1]

z_140_term = solution_140_term[:, 2]
z_145_term = solution_145_term[:, 2]
z_150_term = solution_150_term[:, 2]
z_155_term = solution_155_term[:, 2]
z_160_term = solution_160_term[:, 2]

y_140_term = yw_140_term * y_0
y_145_term = yw_145_term * y_0
y_150_term = yw_150_term * y_0
y_155_term = yw_155_term * y_0
y_160_term = yw_160_term * y_0

x_140_term = y_140_term * (z_140_term + 1)
x_145_term = y_145_term * (z_145_term + 1)
x_150_term = y_150_term * (z_150_term + 1)
x_155_term = y_155_term * (z_155_term + 1)
x_160_term = y_160_term * (z_160_term + 1)

# %%
plt.figure(dpi=300)
plt.semilogy(tau, M1w_160_term)#, label=r'$\tilde{M_1}$')
plt.semilogy(tau, yw_160_term)
# plt.plot(tau, z_130_term)
plt.legend()
plt.show()

# %%
Mn_140_term = y_140_term * (z_140_term + 1)
Mw_140_term = (z_140_term + 2) / (z_140_term + 1) * Mn_140_term

Mn_145_term = y_145_term * (z_145_term + 1)
Mw_145_term = (z_145_term + 2) / (z_145_term + 1) * Mn_145_term

Mn_150_term = y_150_term * (z_150_term + 1)
Mw_150_term = (z_150_term + 2) / (z_150_term + 1) * Mn_150_term

Mn_155_term = y_155_term * (z_155_term + 1)
Mw_155_term = (z_155_term + 2) / (z_155_term + 1) * Mn_155_term

Mn_160_term = y_160_term * (z_160_term + 1)
Mw_160_term = (z_160_term + 2) / (z_160_term + 1) * Mn_160_term

# %% plot curves for termination
kin_curve_125 = np.loadtxt('notebooks/Boyd_kinetic_curves/kinetic_curves/3.txt')
tt_125 = dose2time(kin_curve_125[:, 0] * 1e-6, 1e-9)
L_norm_125 = kin_curve_125[:, 1]

kin_curve_150 = np.loadtxt('notebooks/Boyd_kinetic_curves/kinetic_curves/2.txt')
tt_150 = dose2time(kin_curve_150[:, 0] * 1e-6, 1e-9)
L_norm_150 = kin_curve_150[:, 1]

kin_curve_170 = np.loadtxt('notebooks/Boyd_kinetic_curves/kinetic_curves/1.txt')
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
tau = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mw_125 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mw_125.npy')

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
