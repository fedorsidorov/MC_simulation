import numpy as np
import matplotlib.pyplot as plt
import importlib
from functions import boyd_functions as bf

bf = importlib.reload(bf)

# %%
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

nn = np.arange(1, 50000)
Pn = nn**z_0 * np.exp(-nn / y_0)
Pn /= np.sum(nn**z_0 * np.exp(-nn / y_0))

plt.figure(dpi=300)
plt.semilogx(nn, Pn, '-o')
plt.show()

M1_0 = np.sum(Pn * nn)

# %%
tau_total = 50
tau_step = 0.001
tau = np.arange(0, tau_total, tau_step)

solution_500 = bf.RK4_PCH(500**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_1100 = bf.RK4_PCH(1100**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_2300 = bf.RK4_PCH(2300**-1 * y_0, np.array([1, 1, z_0]), tau)

# %%
M1w_500 = solution_500[:, 0]
M1w_1100 = solution_1100[:, 0]
M1w_2300 = solution_2300[:, 0]

yw_500 = solution_500[:, 1]
yw_1100 = solution_1100[:, 1]
yw_2300 = solution_2300[:, 1]

z_500 = solution_500[:, 2]
z_1100 = solution_1100[:, 2]
z_2300 = solution_2300[:, 2]

y_500 = yw_500 * y_0
y_1100 = yw_1100 * y_0
y_2300 = yw_2300 * y_0

x_500 = y_500 * (z_500 + 1)
x_1100 = y_1100 * (z_1100 + 1)
x_2300 = y_2300 * (z_2300 + 1)

# Pn_500 = nn ** z_500[-1] * np.exp(-nn / y_500[-1])
# Pn_500 /= np.sum(Pn_500)

kin_curve_1 = np.loadtxt('notebooks/odeint/kinetic_curves/1.txt')
tt_1 = kin_curve_1[:, 0] / 20 * 250
L_norm_1 = kin_curve_1[:, 1]

kin_curve_2 = np.loadtxt('notebooks/odeint/kinetic_curves/2.txt')
tt_2 = kin_curve_2[:, 0] / 20 * 250
L_norm_2 = kin_curve_2[:, 1]

kin_curve_3 = np.loadtxt('notebooks/odeint/kinetic_curves/3.txt')
tt_3 = kin_curve_3[:, 0] / 20 * 250
L_norm_3 = kin_curve_3[:, 1]

plt.figure(dpi=300)


plt.plot(tau * 9, M1w_500, label='zip len = 500')
plt.plot(tau * 5, M1w_1100, label='zip len = 1100')
plt.plot(tau * 2.5, M1w_2300, label='zip len = 2300')

plt.plot(tt_3, L_norm_3, 'o--', label='t = 170 C')
plt.plot(tt_2, L_norm_2, 'o--', label='t = 150 C')
plt.plot(tt_1, L_norm_1, 'o--', label='t = 125 C')
plt.xlabel('time, s')
plt.ylabel('L$_{norm}$')
plt.xlim(0, 200)
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

# %
# plt.figure(dpi=300)
# plt.plot([273+140, 273+160, 273+180], [790, 1400, 3200], 'ro')
# plt.show()

