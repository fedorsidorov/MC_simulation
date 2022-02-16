import matplotlib.pyplot as plt
import numpy as np

# %% parameters
Lx = 0.5
Ly = 0.5
lamda = 46
C = 460
rho = 7800
T0 = 20
T_h = 300
T_c = 100

a = lamda / rho / C

# %%
Nx = Ny = N = 100

TT = np.zeros((Nx, Ny))

hx = Lx / (Nx-1)
hy = Ly / (Ny-1)

t_end = 10000
tau = t_end / 100

for i in range(Nx):
    for j in range(Ny):
        TT[i, j] = T0

t = 0

alpha = np.zeros(N)
beta = np.zeros(N)

while t < t_end:
    t += tau

    # Ox
    for j in range(0, Ny):
        alpha[0] = 0
        beta[0] = T_h

        for i in range(1, Nx - 1):
            Ai = lamda / hx**2
            Bi = 2.0 * lamda / hx**2 + rho * C / tau
            Ci = lamda / hx**2
            Fi = -rho * C * TT[i, j] / tau

            alpha[i] = Ai / (Bi - Ci * alpha[i - 1])
            beta[i] = (Ci * beta[i - 1] - Fi) / (Bi - Ci * alpha[i - 1])

        TT[Nx - 1, j] = T_c

        for i in range(Nx - 2, -1, -1):
            TT[i, j] = alpha[i] * TT[i + 1, j] + beta[i]

    # Oy
    for i in range(1, Nx - 1):
        alpha[0] = 2.0 * a * tau / (2.0 * a * tau + hy**2)
        beta[0] = hy**2 * TT[i, 0] / (2.0 * a * tau + hy**2)

        for j in range(1, Ny - 1):
            ai = lamda / hy**2
            bi = 2.0 * lamda / hy**2 + rho * C / tau
            ci = lamda / hy**2
            fi = -rho * C * TT[i, j] / tau
            alpha[j] = ai / (bi - ci * alpha[j - 1])
            beta[j] = (ci * beta[j - 1] - fi) / (bi - ci * alpha[j - 1])

        TT[i, Ny - 1] = (2.0 * a * tau * beta[Ny - 2] + hy**2 * TT[i, Ny - 1]) /\
                        (2.0 * a * tau * (1.0 - alpha[Ny - 2]) + hy**2)

        for j in range(Ny - 2, -1, -1):
            TT[i, j] = alpha[j] * TT[i, j + 1] + beta[j]


# %%
plt.figure(dpi=300)
plt.imshow(TT.transpose())
plt.colorbar()
plt.show()



