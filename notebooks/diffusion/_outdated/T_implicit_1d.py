import matplotlib.pyplot as plt
import numpy as np

# %% parameters
L = 0.1
lamda = 46
C = 460
rho = 7800
T0 = 20
T_l = 300
T_r = 100
delta_t = 60

# %%
N = 100
h = L / (N-1)

TT = np.zeros(N)

t_end = 60
tau = t_end / 100

for i in range(N):
    TT[i] = T0

t = 0

alpha = np.zeros(N)
beta = np.zeros(N)

while t < t_end:
    t += tau

    alpha[0] = 0
    beta[0] = T_l

    for i in range(1, N-1):
        Ai = lamda / h**2
        Bi = 2.0 * lamda / h**2 + rho * C / tau
        Ci = lamda / h**2
        Fi = -rho * C * TT[i] / tau

        alpha[i] = Ai / (Bi - Ci * alpha[i - 1])
        beta[i] = (Ci * beta[i - 1] - Fi) / (Bi - Ci * alpha[i - 1])

    TT[N-1] = T_r

    for i in range(N-2, -1, -1):
        TT[i] = alpha[i] * TT[i + 1] + beta[i]

# %%
plt.figure(dpi=300)
plt.plot(TT)
plt.xlim(0, 100)
plt.ylim(0, 300)
plt.show()



