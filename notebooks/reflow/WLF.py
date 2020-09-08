import numpy as np
import matplotlib.pyplot as plt


# %%
def get_viscosity(T):
    eta_0 = 13450
    T0 = 200
    C1 = 7.6682
    C2 = 210.76
    log_aT = -C1 * (T - T0) / (C2 + (T - T0))
    eta = eta_0 * np.exp(log_aT)
    return eta


# %%
temp = np.linspace(120, 170)

plt.figure(dpi=300)
plt.semilogy(temp, get_viscosity(temp))
plt.show()
