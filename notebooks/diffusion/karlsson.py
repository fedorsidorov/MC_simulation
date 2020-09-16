import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %%
def get_log_D(wp, dT, C_array):
    C1, C2, C3, C4 = C_array
    log_D = wp * dT * C4 + (C1 - wp * C2) + dT * C3
    return log_D


# %%
coefs = np.zeros((4, 4))

coefs[0, :] = -4.428, 1.842, 0, 8.12e-3
coefs[1, :] = 26.0, 37.0, 0.0797, 0
coefs[2, :] = 159.0, 170.0, 0.3664, 0
coefs[3, :] = -13.7, 0.5, 0, 0

# w_pol = np.linspace(0, 1, 100)
delta_T = np.linspace(0, 50, 100)
# delta_T = -34
# delta_T = 0
w_pol = 1
# w_pol = 0.95

plt.figure(dpi=300)

# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[0, :]), label='1')
# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[1, :]), label='2')
# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[2, :]), label='3')
# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[3, :]), label='4')

# plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[0, :]), label='1')
plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[1, :]), label='2')  # region of interest
# plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[2, :]), label='3')
# plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[3, :]), label='4')

# plt.xlim(0.75, 1)
# plt.ylim(-15, 0)

plt.grid()
plt.legend()

plt.show()

