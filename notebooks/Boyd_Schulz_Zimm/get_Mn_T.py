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
zip_len_term_147, _ = bf.get_zip_len_term_trans(147)
zip_len_term_148, _ = bf.get_zip_len_term_trans(148)
zip_len_term_149, _ = bf.get_zip_len_term_trans(149)
# zip_len_term_150, _ = bf.get_zip_len_term_trans(150)
zip_len_term_151, _ = bf.get_zip_len_term_trans(151)
zip_len_term_152, _ = bf.get_zip_len_term_trans(152)
zip_len_term_153, _ = bf.get_zip_len_term_trans(153)


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

solution_147_term = bf.RK4_PCH(zip_len_term_147**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_148_term = bf.RK4_PCH(zip_len_term_148**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_149_term = bf.RK4_PCH(zip_len_term_149**-1 * y_0, np.array([1, 1, z_0]), tau)
# solution_150_term = bf.RK4_PCH(zip_len_term_150**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_151_term = bf.RK4_PCH(zip_len_term_151**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_152_term = bf.RK4_PCH(zip_len_term_152**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_153_term = bf.RK4_PCH(zip_len_term_153**-1 * y_0, np.array([1, 1, z_0]), tau)


# %%
M1w_147_term = solution_147_term[:, 0]
M1w_148_term = solution_148_term[:, 0]
M1w_149_term = solution_149_term[:, 0]
# M1w_150_term = solution_150_term[:, 0]
M1w_151_term = solution_151_term[:, 0]
M1w_152_term = solution_152_term[:, 0]
M1w_153_term = solution_153_term[:, 0]

yw_147_term = solution_147_term[:, 1]
yw_148_term = solution_148_term[:, 1]
yw_149_term = solution_149_term[:, 1]
# yw_150_term = solution_150_term[:, 1]
yw_151_term = solution_151_term[:, 1]
yw_152_term = solution_152_term[:, 1]
yw_153_term = solution_153_term[:, 1]

z_147_term = solution_147_term[:, 2]
z_148_term = solution_148_term[:, 2]
z_149_term = solution_149_term[:, 2]
# z_150_term = solution_150_term[:, 2]
z_151_term = solution_151_term[:, 2]
z_152_term = solution_152_term[:, 2]
z_153_term = solution_153_term[:, 2]

y_147_term = yw_147_term * y_0
y_148_term = yw_148_term * y_0
y_149_term = yw_149_term * y_0
# y_150_term = yw_150_term * y_0
y_151_term = yw_151_term * y_0
y_152_term = yw_152_term * y_0
y_153_term = yw_153_term * y_0

x_147_term = y_147_term * (z_147_term + 1)
x_148_term = y_148_term * (z_148_term + 1)
x_149_term = y_149_term * (z_149_term + 1)
# x_150_term = y_150_term * (z_150_term + 1)
x_151_term = y_151_term * (z_151_term + 1)
x_152_term = y_152_term * (z_152_term + 1)
x_153_term = y_153_term * (z_153_term + 1)


# %%
# plt.figure(dpi=300)
# plt.semilogy(tau, M1w_160_term)#, label=r'$\tilde{M_1}$')
# plt.semilogy(tau, yw_160_term)
# plt.plot(tau, z_130_term)
# plt.legend()
# plt.show()

# %%
Mn_147_term = y_147_term * (z_147_term + 1)
Mw_147_term = (z_147_term + 2) / (z_147_term + 1) * Mn_147_term

Mn_148_term = y_148_term * (z_148_term + 1)
Mw_148_term = (z_148_term + 2) / (z_148_term + 1) * Mn_148_term

Mn_149_term = y_149_term * (z_149_term + 1)
Mw_149_term = (z_149_term + 2) / (z_149_term + 1) * Mn_149_term

# Mn_150_term = y_150_term * (z_150_term + 1)
# Mw_150_term = (z_150_term + 2) / (z_150_term + 1) * Mn_150_term

Mn_151_term = y_151_term * (z_151_term + 1)
Mw_151_term = (z_151_term + 2) / (z_151_term + 1) * Mn_151_term

Mn_152_term = y_152_term * (z_152_term + 1)
Mw_152_term = (z_152_term + 2) / (z_152_term + 1) * Mn_152_term

Mn_153_term = y_153_term * (z_153_term + 1)
Mw_153_term = (z_153_term + 2) / (z_153_term + 1) * Mn_153_term


# %%
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_147_term.npy', Mn_147_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_147_term.npy', Mw_147_term)

np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_148_term.npy', Mn_148_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_148_term.npy', Mw_148_term)

np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_149_term.npy', Mn_149_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_149_term.npy', Mw_149_term)

# np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_150_term.npy', Mn_150_term)
# np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_150_term.npy', Mw_150_term)

np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_151_term.npy', Mn_151_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_151_term.npy', Mw_151_term)

np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_152_term.npy', Mn_152_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_152_term.npy', Mw_152_term)

np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_153_term.npy', Mn_153_term)
np.save('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mw_153_term.npy', Mw_153_term)


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
