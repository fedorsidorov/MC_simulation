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
zip_len_term_98, zip_len_trans_98 = bf.get_zip_len_term_trans(98)
zip_len_term_118, zip_len_trans_118 = bf.get_zip_len_term_trans(118)
zip_len_term_125, zip_len_trans_125 = bf.get_zip_len_term_trans(125)
zip_len_term_150, zip_len_trans_150 = bf.get_zip_len_term_trans(150)
zip_len_term_170, zip_len_trans_170 = bf.get_zip_len_term_trans(170)

# %% 2-point curves
L_norm = np.array([1, 0.5, 0])

doses_118_1 = np.array([0, 3.8, 30])
doses_98_6 = np.array([0, 22, 180])
doses_125_6 = np.array([0, 5.2, 46])
doses_125_0p15 = np.array([0, 0.85, 8])

doses_125_150nm = np.array([0, 1.3, 10.2])
doses_125_1um = np.array([0, 20, 170])
doses_150_1um = np.array([0, 3.6, 30])
doses_170_2um = np.array([0, 4.0, 33.6])

# %%
tau_total = 400
tau_step = 0.01
tau = np.arange(0, tau_total, tau_step)

solution_98_term = bf.RK4_PCH(zip_len_term_98**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_118_term = bf.RK4_PCH(zip_len_term_118**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_125_term = bf.RK4_PCH(zip_len_term_125**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_150_term = bf.RK4_PCH(zip_len_term_150**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_170_term = bf.RK4_PCH(zip_len_term_170**-1 * y_0, np.array([1, 1, z_0]), tau)

solution_98_trans = bf.RK4_PCH(zip_len_trans_98**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_118_trans = bf.RK4_PCH(zip_len_trans_118**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_125_trans = bf.RK4_PCH(zip_len_trans_125**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_150_trans = bf.RK4_PCH(zip_len_trans_150**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_170_trans = bf.RK4_PCH(zip_len_trans_170**-1 * y_0, np.array([1, 1, z_0]), tau)

# %%
M1w_98_term = solution_98_term[:, 0]
M1w_118_term = solution_118_term[:, 0]
M1w_125_term = solution_125_term[:, 0]
M1w_150_term = solution_150_term[:, 0]
M1w_170_term = solution_170_term[:, 0]

yw_98_term = solution_98_term[:, 1]
yw_118_term = solution_118_term[:, 1]
yw_125_term = solution_125_term[:, 1]
yw_150_term = solution_150_term[:, 1]
yw_170_term = solution_170_term[:, 1]

z_98_term = solution_98_term[:, 2]
z_118_term = solution_118_term[:, 2]
z_125_term = solution_125_term[:, 2]
z_150_term = solution_150_term[:, 2]
z_170_term = solution_170_term[:, 2]

y_98_term = yw_98_term * y_0
y_118_term = yw_118_term * y_0
y_125_term = yw_125_term * y_0
y_150_term = yw_150_term * y_0
y_170_term = yw_170_term * y_0

x_98_term = y_98_term * (z_125_term + 1)
x_118_term = y_118_term * (z_125_term + 1)
x_125_term = y_125_term * (z_125_term + 1)
x_150_term = y_150_term * (z_150_term + 1)
x_170_term = y_170_term * (z_170_term + 1)

M1w_98_trans = solution_98_trans[:, 0]
M1w_118_trans = solution_118_trans[:, 0]
M1w_125_trans = solution_125_trans[:, 0]
M1w_150_trans = solution_150_trans[:, 0]
M1w_170_trans = solution_170_trans[:, 0]

yw_98_trans = solution_98_trans[:, 1]
yw_118_trans = solution_118_trans[:, 1]
yw_125_trans = solution_125_trans[:, 1]
yw_150_trans = solution_150_trans[:, 1]
yw_170_trans = solution_170_trans[:, 1]

z_98_trans = solution_98_trans[:, 2]
z_118_trans = solution_118_trans[:, 2]
z_125_trans = solution_125_trans[:, 2]
z_150_trans = solution_150_trans[:, 2]
z_170_trans = solution_170_trans[:, 2]

y_98_trans = yw_98_trans * y_0
y_118_trans = yw_118_trans * y_0
y_125_trans = yw_125_trans * y_0
y_150_trans = yw_150_trans * y_0
y_170_trans = yw_170_trans * y_0

x_98_trans = y_98_trans * (z_125_trans + 1)
x_118_trans = y_118_trans * (z_125_trans + 1)
x_125_trans = y_125_trans * (z_125_trans + 1)
x_150_trans = y_150_trans * (z_150_trans + 1)
x_170_trans = y_170_trans * (z_170_trans + 1)

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

font_size = 8

_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
fig.set_size_inches(4, 3)

# plt.plot(tau * 1, M1w_98_term, label='sim T = 98 C°')
# plt.plot(tau * 0.5, M1w_118_term, label='sim T = 118 C°')

plt.plot(tau * 8.5, M1w_125_term, label='модел. T = 125 C°')
plt.plot(tau * 4, M1w_150_term, label='модел. T = 150 C°')
plt.plot(tau * 2.5, M1w_170_term, label='модел. T = 170 C°')

# plt.plot(tau * 8.5, M1w_125_term, label='sim T = 125 C°')
# plt.plot(tau * 4, M1w_150_term, label='sim T = 150 C°')

# plt.plot(dose2time(doses_98_6 * 1e-6, 6e-9), L_norm, 'o', label='exp T = 98 C°, 6 nA')
# plt.plot(dose2time(doses_118_1 * 1e-6, 6e-9), L_norm, 'o', label='exp T = 118 C°, 6 nA')

plt.plot(tt_125, L_norm_125, 'o', label='эксп. T = 125 C°')
plt.plot(tt_150, L_norm_150, 'o', label='эксп. T = 150 C°')
plt.plot(tt_170, L_norm_170, 'o', label='эксп. T = 170 C°')

# plt.plot(dose2time(doses_125_1um * 1e-6, 1e-9), L_norm, 'o', label='exp T = 125 C°, 1 nA, 1000 nm')
# plt.plot(dose2time(doses_150_1um * 1e-6, 1e-9), L_norm, 'o', label='exp T = 150 C°, 1 nA, 1000 nm')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.xlim(0, 10)

plt.legend()
plt.legend(fontsize=font_size)

plt.title('d$_{PMMA}$ = 80 нм, 200 c - 16 мкКл/см$^2$')
plt.xlabel('t, с', fontsize=font_size)
plt.ylabel('L$_{norm}$', fontsize=font_size)

plt.xlim(0, 200)
plt.ylim(0, 1)
plt.grid()
plt.show()

# plt.savefig('for_report.tiff', bbox_inches='tight')

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
plt.plot(tau, Mw_125)
plt.grid()
plt.show()

