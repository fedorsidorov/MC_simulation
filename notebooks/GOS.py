import importlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
import constants as c
import grid as g
import utilities as u

u = importlib.reload(u)
c = importlib.reload(c)
g = importlib.reload(g)


# %%
def get_df_dW_1s(k, hw_eV, Zs):
    n = 1  # K shell

    Q_0 = c.hbar ** 2 * k ** 2 / (2 * c.m)
    W_0 = hw_eV * c.eV

    Q = Q_0 / (Zs ** 2 * c.Ry)
    W = W_0 / (Zs ** 2 * c.Ry)

    k2 = W - 1 / n ** 2

    def A(QQ, WW):

        if k2 > 0:
            kappa = np.sqrt(k2)
            fC = (1 - np.exp(-2 * np.pi / kappa)) ** (-1)

            return 2 ** 5 * (2 / n) ** 3 * WW * np.exp(
                -2 / kappa * np.arctan(
                    2 * kappa / n / (QQ - WW + 2 / n ** 2)
                )
            ) * fC

        else:
            return 2 ** 5 * (2 / n) ** 3 * WW * np.exp(
                -1 / np.sqrt(-k2) * np.log(
                    (QQ - WW + 2 / n ** 2 + 2 * np.sqrt(-k2) / n) / (QQ - WW + 2 / n ** 2 - 2 * np.sqrt(-k2) / n)
                )
            )

    def B(QQ, WW):
        return ((QQ - WW) ** 2 + (2 / n) ** 2 * QQ) ** (-2 * (n + 1))

    def C(QQ, WW):
        return QQ + WW / 3

    df_dW = A(Q, W) * B(Q, W) * C(Q, W)

    df_dW_0 = 1 / (Zs ** 2 * c.Ry) * df_dW

    return df_dW_0


def get_df_dW_K(k, hw_eV, Zs):  # BOOK
    E = hw_eV*c.eV
    Qp = (k*c.a0/Zs)**2
    kH2 = hw_eV*c.eV / (Zs**2 * c.Ry) - 1

    if kH2 > 0:
        kH = np.sqrt(kH2)
        tan_beta_p = 2*kH / (Qp - kH2 + 1)
        beta_p = np.arctan(tan_beta_p)

        if beta_p < 0:  # seems to be important
            beta_p += np.pi

        num = 256*E*(Qp + kH2**2 / 3 + 1/3)*np.exp(-2*beta_p/kH)
        den = Zs**4 * c.Ry**2 * ((Qp - kH2 + 1)**2 + 4*kH2)**3 * (1 - np.exp(-2*np.pi/kH))

    else:
        y = -(-kH2)**(-1/2) * np.log((Qp + 1 - kH2 + 2*(-kH2)**(1/2))/(Qp + 1 - kH2 - 2*(-kH2)**(1/2)))
        num = 256*E*(Qp + kH2/3 + 1/3)*np.exp(y)
        den = Zs**4 * c.Ry**2 * ((Qp - kH2 + 1)**2 + 4*kH2)**3

    return num / den


#%% test GOS for C K-shell
EE = np.linspace(300, 1200, 50)  # eV
kk = np.linspace(0.01, 100, 500)  # inv A
FF = np.zeros((len(EE), len(kk)))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

for i, Ei in enumerate(EE):
    for j, ki in enumerate(kk):
        FF[i, j] = get_df_dW_K(ki*1e+8, Ei, c.Zs_C) * c.eV * 1e+3

    ax.plot(np.ones(len(kk))*EE[i], np.log((kk*1e+8*c.a0)**2), FF[i, :])

ax.view_init(30, 30)
plt.grid()
ax.set_xlabel('E, eV')
ax.set_ylabel('ln(ka$_0$)')
ax.set_zlabel('df/dW * 10$^3$, eV$^{-1}$')
plt.show()

plt.savefig('GOS.png', dpi=300)

#%%
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(1)


# %%
def get_PMMA_ELG_GOS(q, hw_eV):
    w = hw_eV * c.eV / c.hbar
    k = q / c.hbar

    C_alpha, O_alpha = c.N_C_MMA, c.N_O_MMA
    C_Zs, O_Zs = 5.7, 7.7

    ELF = 2 * np.pi ** 2 * c.n_MMA / w * C_alpha * get_df_dw_1s(k, w, C_Zs) * O_alpha * get_df_dw_1s(k, w, O_Zs)

    return ELF


def get_PMMA_DIIMFP_GOS(E_eV, hw_eV):
    if hw_eV > E_eV:
        return 0

    E = E_eV * c.eV
    hw = hw_eV * c.eV

    def get_Y(k):
        return get_PMMA_ELG_GOS(k * c.hbar, hw_eV) / k

    km, kp = u.get_km_kp(E, hw)
    integral = integrate.quad(get_Y, km, kp)[0]

    return 1 / (np.pi * c.a0 * E_eV) * integral  # cm^-1 * eV^-1


# %% test optical PMMA ELF
EE = g.EE
OLF = np.zeros(len(EE))

for i, Ei in enumerate(EE):
    OLF[i] = get_PMMA_ELG_GOS(0, Ei)

plt.figure(dpi=300)
plt.loglog(EE, OLF)
plt.show()
