import numpy as np
import matplotlib.pyplot as plt
import importlib
from tqdm import tqdm
from functions import boyd_functions as bf

bf = importlib.reload(bf)

# %% 80 nm, 125 C
n_monomers = 22910324
n_scissions = 219914

D = 20  # uC/cm^2
total_time = 250  # s

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

nn = np.arange(1, 50000)
Pn = nn**z_0 * np.exp(-nn / y_0)
Pn /= np.sum(Pn)  # norm it

M1_0 = np.sum(Pn * nn)

# plt.figure(dpi=300)
# plt.semilogx(nn, Pn, '-o')
# plt.show()

# from Boyd simulation%
# t = alpha * tau == tau /  (y0 * k_s)
alpha = 8.5

k_s_sim = 1 / alpha / y_0
k_s_exp = n_scissions / n_monomers / total_time

# %%
tau_total = 400
tau_step = 0.01
tau = np.arange(0, tau_total, tau_step)

zip_len_term_125, _ = bf.get_zip_len_term_trans(125)

solution_125 = bf.RK4_PCH(zip_len_term_125**-1 * y_0, np.array([1, 1, z_0]), tau)

M1w_125 = solution_125[:, 0]
yw_125 = solution_125[:, 1]
z_125 = solution_125[:, 2]
y_125 = yw_125 * y_0
x_125 = y_125 * (z_125 + 1)

M1 = M1w_125 * M1_0

# %%
# Pn_evolution = np.zeros((len(tau), len(Pn)))
#
# for i in range(len(tau)):
#     Pn_evolution[i, :] = nn**z_125[i] * np.exp(-nn / y_125[i])

# %%
Mn_125 = y_125 * (z_125 + 1)
Mw_125 = (z_125 + 2) / (z_125 + 1) * Mn_125

Mn_125_test = np.zeros(len(Mn_125))
Mw_125_test = np.zeros(len(Mn_125))

progress_bar = tqdm(total=len(Mn_125_test), position=0)

for i in range(len(Mn_125_test)):
    Pn_now = nn ** z_125[i] * np.exp(-nn / y_125[i])
    Pn_now /= np.sum(Pn_now)  # norm it

    Mn_125_test[i] = np.sum(Pn_now * nn)
    Mw_125_test[i] = np.sum(Pn_now * nn**2) / np.sum(Pn_now * nn)

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.semilogy(Mn_125, label='Mn')
plt.semilogy(Mn_125_test, '--', label='Mn test')

plt.semilogy(Mw_125, label='Mw')
plt.semilogy(Mw_125_test, '--', label='Mw test')

plt.grid()
plt.legend()

plt.show()

# %%
np.save('notebooks/Boyd_kinetic_curves/arrays_Si/tau.npy', tau)
np.save('notebooks/Boyd_kinetic_curves/arrays_Si/Mw_125.npy', Mw_125)
