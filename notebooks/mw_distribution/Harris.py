import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def shulz_zimm_M(M, beta, z):  # miyake1960.pdf
    A = beta ** (z + 1) / gamma(z + 1)
    return A * M ** z * np.exp(-beta * M)


def get_beta_z(Mn, Mw):  # miyake1960.pdf
    beta = 1 / (Mw - Mn)
    z = Mn / (Mw - Mn)
    return beta, z


# %%
# Harris
mw = np.arange(100, 2e+7 + 0.1, 100)
Mn = 5.63e+5
Mw = 2.26e+6

mw_distr_M = shulz_zimm_M(mw, *get_beta_z(Mn, Mw))
mw_probs_M = mw_distr_M / np.sum(mw_distr_M)
mw_probs_N = (mw_distr_M / mw) / np.sum(mw_distr_M / mw)

# plt.figure(dpi=300)
# plt.semilogx(mw[::1000], mw_distr_M[::1000])
# plt.semilogx(mw[::1000], mw_probs_N[::1000])
# plt.semilogx(mw[::1000], mw_probs_m[::1000], '.-')
# plt.show()

# %% check
beta, z = get_beta_z(Mn, Mw)

# Mn_check = z / beta
# Mw_check = (z + 1) / z * Mn_check

Mn_check = 1 / np.sum(1 / mw * mw_distr_M)
Mw_check = np.sum(mw * mw_distr_M)

# %% integral
mw_distr_M_integral = np.ones(len(mw))

for i in range(len(mw) - 1):
    mw_distr_M_integral[i] = np.sum(mw_probs_M[:i+1])

# %%
harris = np.loadtxt('notebooks/mw_distribution/Harris_curves/Harris_integral_initial.txt')

plt.figure(dpi=300)

plt.semilogx(harris[:, 0], harris[:, 1])

plt.semilogx(mw, mw_distr_M_integral)

plt.show()

