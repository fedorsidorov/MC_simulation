import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def shulz_zimm(M, beta, z):  # miyake1960.pdf
    A = beta ** (z + 1) / gamma(z + 1)
    return A * M ** z * np.exp(-beta * M)


def get_beta_z(Mn, Mw):  # miyake1960.pdf
    beta = 1 / (Mw - Mn)
    z = Mn / (Mw - Mn)
    return beta, z


# %%
# Harris
mw = np.arange(100, 2e+7 + 0.1, 1)
Mn = 5.63e+5
Mw = 2.26e+6

# 950K
# mw = np.arange(100, 4e+6 + 0.1, 1)
# mw = np.logspace(2, 6.6, 1000)
# Mn = 271374.7
# Mw = 669184.4

mw_distr = shulz_zimm(mw, *get_beta_z(Mn, Mw))
mw_probs_m = mw_distr / np.sum(mw_distr)
mw_probs_n = (mw_distr / mw) / np.sum(mw_distr / mw)

plt.figure(dpi=300)
# plt.semilogx(mw[::1000], mw_distr[::1000])
# plt.semilogx(mw[::1000], mw_probs_n[::1000])
plt.semilogx(mw[::1000], mw_probs_m[::1000], '.-')
plt.show()

# %%
n_chains = 10000
weights_m = np.zeros(n_chains)
weights_n = np.zeros(n_chains)

progress_bar = tqdm(total=n_chains, position=0)

for i_m in range(n_chains):
    weights_m[i_m] = np.random.choice(mw, p=mw_probs_m)
    weights_n[i_m] = np.random.choice(mw, p=mw_probs_n)
    progress_bar.update()

# %%
weights_unique_m = np.zeros((len(np.unique(weights_m)), 2))
weights_unique_n = np.zeros((len(np.unique(weights_n)), 2))

for i_m, weight_unique_m in enumerate(np.unique(weights_m)):
    weights_unique_m[i_m, 0] = weight_unique_m
    weights_unique_m[i_m, 1] = len(np.where(weights_m == weight_unique_m)[0])

for i_n, weight_unique_n in enumerate(np.unique(weights_n)):
    weights_unique_n[i_n, 0] = weight_unique_n
    weights_unique_n[i_n, 1] = len(np.where(weights_n == weight_unique_n)[0])

# %%
Mi = weights_unique_n[:, 0]
Ni = weights_unique_n[:, 1]

Mn_n = np.sum(Ni * Mi) / np.sum(Ni)
Mw_n = np.sum(Ni * Mi ** 2) / np.sum(Ni * Mi)
