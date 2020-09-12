import numpy as np
import matplotlib.pyplot as plt


# %%
def get_viscosity(T):  # aho2008.pdf
    eta_0 = 13450
    T0 = 200
    C1 = 7.6682
    C2 = 210.76
    log_aT = -C1 * (T - T0) / (C2 + (T - T0))
    eta = eta_0 * np.exp(log_aT)
    return eta


def get_viscosity_W(T, Mw):  # aho2008.pdf, bueche1955.pdf
    Mw_0 = 9e+4
    eta = get_viscosity(T)
    eta_final = eta * (Mw / Mw_0)**3.4
    return eta_final


# %%
temp = np.linspace(120, 170)

plt.figure(dpi=300)
plt.semilogy(temp, get_viscosity_W(temp, 80e+3))
plt.show()
