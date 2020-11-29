import numpy as np
import matplotlib.pyplot as plt


# %%
Ln = 2714
Lw = 6692

x = Ln
z = (2 - Lw/Ln) / (Lw/Ln - 1)
y = x / (z + 1)

nn = np.arange(2, 100000)

C0 = 1 / np.sum(nn**z * np.exp(-nn / y))


def get_Pn():
    return C0 * nn**z * np.exp(-nn / y)


def get_moment(i):
    return np.sum(nn**i * C0 * nn**z * np.exp(-nn / y))


def get_moment_int(i):
    return np.trapz(nn**i * C0 * nn**z * np.exp(-nn / y), x=nn)


M0 = 0
M1 = 0
M2 = 0

Pn = get_Pn()

# plt.figure(dpi=300)
# plt.semilogx(nn, Pn)
# plt.show()
