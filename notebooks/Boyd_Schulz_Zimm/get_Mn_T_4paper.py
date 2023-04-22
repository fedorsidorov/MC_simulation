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
TT = [120, 140, 160, 180]

zip_len_term_100, zip_len_trans_100 = bf.get_zip_len_term_trans(100)
zip_len_term_120, zip_len_trans_120 = bf.get_zip_len_term_trans(120)
zip_len_term_140, zip_len_trans_140 = bf.get_zip_len_term_trans(140)
zip_len_term_150, zip_len_trans_150 = bf.get_zip_len_term_trans(150)
zip_len_term_160, zip_len_trans_160 = bf.get_zip_len_term_trans(160)
zip_len_term_180, zip_len_trans_180 = bf.get_zip_len_term_trans(180)
zip_len_term_200, zip_len_trans_200 = bf.get_zip_len_term_trans(200)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(TT, [zip_len_term_120, zip_len_term_140, zip_len_term_160, zip_len_term_180], '.--', label='termination')
    ax.plot(TT, [zip_len_trans_120, zip_len_trans_140, zip_len_trans_160, zip_len_trans_180], '.--', label='transfer')

    ax.legend(loc=0)
    ax.set(xlabel=r'$T$, $^\circ$C')
    ax.set(ylabel=r'zip length')
    ax.autoscale(tight=True)

    plt.xlim(100, 200)
    plt.ylim(0, 4000)

    fig.savefig('zip_len_term_trans.jpg', dpi=600)
    plt.show()


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

solution_100_term = bf.RK4_PCH(zip_len_term_100**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_120_term = bf.RK4_PCH(zip_len_term_120**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_140_term = bf.RK4_PCH(zip_len_term_140**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_150_term = bf.RK4_PCH(zip_len_term_150**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_160_term = bf.RK4_PCH(zip_len_term_160**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_180_term = bf.RK4_PCH(zip_len_term_180**-1 * y_0, np.array([1, 1, z_0]), tau)
solution_200_term = bf.RK4_PCH(zip_len_term_200**-1 * y_0, np.array([1, 1, z_0]), tau)


# %%
M1w_100_term = solution_100_term[:, 0]
M1w_120_term = solution_120_term[:, 0]
M1w_140_term = solution_140_term[:, 0]
M1w_150_term = solution_150_term[:, 0]
M1w_160_term = solution_160_term[:, 0]
M1w_180_term = solution_180_term[:, 0]
M1w_200_term = solution_200_term[:, 0]

yw_100_term = solution_100_term[:, 1]
yw_120_term = solution_120_term[:, 1]
yw_140_term = solution_140_term[:, 1]
yw_150_term = solution_150_term[:, 1]
yw_160_term = solution_160_term[:, 1]
yw_180_term = solution_180_term[:, 1]
yw_200_term = solution_200_term[:, 1]

z_100_term = solution_100_term[:, 2]
z_120_term = solution_120_term[:, 2]
z_140_term = solution_140_term[:, 2]
z_150_term = solution_150_term[:, 2]
z_160_term = solution_160_term[:, 2]
z_180_term = solution_180_term[:, 2]
z_200_term = solution_200_term[:, 2]

y_100_term = yw_100_term * y_0
y_120_term = yw_120_term * y_0
y_140_term = yw_140_term * y_0
y_150_term = yw_150_term * y_0
y_160_term = yw_160_term * y_0
y_180_term = yw_180_term * y_0
y_200_term = yw_200_term * y_0

x_100_term = y_100_term * (z_100_term + 1)
x_120_term = y_120_term * (z_120_term + 1)
z_140_term = z_140_term * (z_140_term + 1)
z_150_term = z_150_term * (z_150_term + 1)
z_160_term = z_160_term * (z_160_term + 1)
z_180_term = z_180_term * (z_180_term + 1)
x_200_term = y_200_term * (z_200_term + 1)

# %%
# plt.figure(dpi=300)
# plt.semilogy(tau, M1w_160_term)#, label=r'$\tilde{M_1}$')
# plt.semilogy(tau, yw_160_term)
# plt.plot(tau, z_130_term)
# plt.legend()
# plt.show()

# %%
Mn_100_term = y_100_term * (z_100_term + 1)
Mw_100_term = (z_100_term + 2) / (z_100_term + 1) * Mn_100_term

Mn_120_term = y_120_term * (z_120_term + 1)
Mw_120_term = (z_120_term + 2) / (z_120_term + 1) * Mn_120_term

Mn_140_term = y_140_term * (z_140_term + 1)
Mw_140_term = (z_140_term + 2) / (z_140_term + 1) * Mn_140_term

Mn_150_term = y_150_term * (z_150_term + 1)
Mw_150_term = (z_150_term + 2) / (z_150_term + 1) * Mn_150_term

Mn_160_term = y_160_term * (z_160_term + 1)
Mw_160_term = (z_160_term + 2) / (z_160_term + 1) * Mn_160_term

Mn_180_term = y_180_term * (z_180_term + 1)
Mw_180_term = (z_180_term + 2) / (z_180_term + 1) * Mn_180_term

Mn_200_term = y_200_term * (z_200_term + 1)
Mw_200_term = (z_200_term + 2) / (z_200_term + 1) * Mn_200_term


# %%
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mn_100_term.npy', Mn_100_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_100_term.npy', Mw_100_term)

np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_120_term.npy', Mw_120_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_120_term.npy', Mw_120_term)

np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mn_140_term.npy', Mn_140_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_140_term.npy', Mw_140_term)

np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mn_150_term.npy', Mn_150_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_150_term.npy', Mw_150_term)

np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mn_160_term.npy', Mn_160_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_160_term.npy', Mw_160_term)

np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mn_180_term.npy', Mn_180_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_180_term.npy', Mw_180_term)

np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mn_200_term.npy', Mn_200_term)
np.save('notebooks/Boyd_Schulz_Zimm/4paper/Mw_200_term.npy', Mw_200_term)

# %%
tau = np.load('notebooks/Boyd_Schulz_Zimm/arrays/tau.npy')

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogy(tau, Mn_100_term * 100, label=r'100 $^\circ$C')
    ax.semilogy(tau, Mn_150_term * 100, label=r'150 $^\circ$C')
    ax.semilogy(tau, Mn_200_term * 100, label=r'200 $^\circ$C')

    # ax.semilogy(tau, Mw_100_term * 100, label=r'100 $^\circ$C')
    # ax.semilogy(tau, Mw_150_term * 100, label=r'150 $^\circ$C')
    # ax.semilogy(tau, Mw_200_term * 100, label=r'200 $^\circ$C')

    ax.legend(loc=1)
    ax.set(xlabel=r'$\tau$')
    # ax.set(ylabel=r'$M_n$')
    ax.set(ylabel=r'$M_w$')
    ax.autoscale(tight=True)

    # plt.xlim(10, 1e+4)
    plt.ylim(1e+1 * 100, 1e+4 * 100)

    fig.savefig('Mn_tau_100_150_200.jpg', dpi=600)
    # fig.savefig('Mw_tau_100_150_200.jpg', dpi=600)
    # plt.show()


