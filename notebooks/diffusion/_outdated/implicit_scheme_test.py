import matplotlib.pyplot as plt
import numpy as np

# %% parameters
L = 0.1
lamda = 46
C = 460
rho = 7800
T0 = 20
delta_t = 60

# %% 1st boundary condition
N = 100
h = L / (N-1)

T_l = 300
T_r = 100

TT = np.ones(N) * T0

t_end = 50
tau = t_end / 100

t = 0

alphas = np.zeros(N)
betas = np.zeros(N)

while t < t_end:
    t += tau

    alphas[0] = 0
    betas[0] = T_l

    for i in range(1, N-1):
        Ai = lamda / h**2
        Bi = 2 * lamda / h**2 + rho * C / tau
        Ci = lamda / h**2
        Fi = -rho * C * TT[i] / tau

        alphas[i] = Ai / (Bi - Ci * alphas[i - 1])
        betas[i] = (Ci * betas[i - 1] - Fi) / (Bi - Ci * alphas[i - 1])

    TT[N-1] = T_r

    for i in range(N-2, -1, -1):
        TT[i] = alphas[i] * TT[i + 1] + betas[i]

# %
plt.figure(dpi=300)
plt.plot(TT)
plt.show()

# %% 2nd boundary condition, q = 0
N = 100
h = L / (N-1)

t_end = 60
tau = t_end / 100

TT = np.ones(N) * T0
TT[40:60] = 40

t = 0

alphas = np.zeros(N)
betas = np.zeros(N)

a = lamda / rho / C

while t < t_end:
    t += tau

    alphas[0] = 2 * a * tau / (h**2 + 2 * a * tau)
    betas[0] = h**2 * TT[0] / (h**2 + 2 * a * tau)

    for i in range(1, N-1):
        Ai = lamda / h**2
        Bi = 2 * lamda / h**2 + rho * C / tau
        Ci = lamda / h**2
        Fi = -rho * C * TT[i] / tau

        alphas[i] = Ai / (Bi - Ci * alphas[i - 1])
        betas[i] = (Ci * betas[i - 1] - Fi) / (Bi - Ci * alphas[i - 1])

    TT[N-1] = (2 * a * tau * lamda * betas[N-2] + h**2 * lamda * TT[N-1]) /\
              (lamda * h**2 + 2 * a * tau * lamda * (1 - alphas[N-2]))

    # TT[N - 1] = T_r

    for i in range(N-2, -1, -1):
        TT[i] = alphas[i] * TT[i + 1] + betas[i]

# %
plt.figure(dpi=300)
plt.plot(TT)
plt.show()




