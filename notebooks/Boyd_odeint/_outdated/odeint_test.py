import numpy as np
from scipy. integrate import odeint, RK45
import matplotlib.pyplot as plt


def model_general(yw_z_M1w, t):

    alpha = 0.1
    gw = alpha / (10 + 1)

    yw, z, M1w = yw_z_M1w[0], yw_z_M1w[1], yw_z_M1w[2]

    A = -(1 / (yw * (1 + gw * yw * (z + 1))) +
          (z + 2) * gw / ((z + 2) * gw * yw + 2))

    B = -(1 / ((z + 1) * (1 + gw * yw * (z + 1))) +
          gw * yw / ((z + 2) * gw * yw + 2))

    C = yw * (z + 1) / (gw * yw * (z + 1) + 1) -\
        (1 / 3 * (z + 2) * (z + 3) * gw * yw**2 + 2 * (z + 2) * yw) /\
        ((z + 2) * gw * yw + 2)

    D = (z + 2) * gw / ((z + 2) * gw * yw + 2) -\
        (2 * (z + 2) * (z + 3) * gw * yw + 3 * (z + 2)) /\
        ((z + 2) * (z + 3) * gw * yw**2 + 3 * (z + 2) * yw)

    E = gw * yw / ((z + 2) * gw * yw + 2) -\
        (gw * yw * (2 * z + 5) + 3) / ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))

    F = (1 / 3 * (z + 2) * (z + 3) * gw * yw**2 + 2 * (z + 2) * yw) / ((z + 2) * gw * yw + 2) -\
        1 / 2 * (z + 2) * (z + 3) * (z + 4) * (gw * yw**2 + 3 * (z + 2) * (z + 3) * yw) /\
        ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))

    dyw_dt = (B * F - C * E) / (A * E - D * B)
    dz_dt = (C * D - A * F) / (A * E - D * B)

    dM1w_dt = M1w * (1 / yw * dyw_dt + 1 / (z + 1) * dz_dt - yw * (z + 1)) / (1 + gw * yw * (z + 1))

    return [dyw_dt, dz_dt, dM1w_dt]


def model_general_cut(t, yw_z):

    z0 = 10
    alpha = 0.1
    gw = alpha / (z0 + 1)

    yw, z = yw_z[0], yw_z[1]

    A = -(1 / (yw * (1 + gw * yw * (z + 1))) +
          (z + 2) * gw / ((z + 2) * gw * yw + 2))

    B = -(1 / ((z + 1) * (1 + gw * yw * (z + 1))) +
          gw * yw / ((z + 2) * gw * yw + 2))

    C = yw * (z + 1) / (gw * yw * (z + 1) + 1) -\
        (1 / 3 * (z + 2) * (z + 3) * gw * yw**2 + 2 * (z + 2) * yw) /\
        ((z + 2) * gw * yw + 2)

    D = (z + 2) * gw / ((z + 2) * gw * yw + 2) -\
        (2 * (z + 2) * (z + 3) * gw * yw + 3 * (z + 2)) /\
        ((z + 2) * (z + 3) * gw * yw**2 + 3 * (z + 2) * yw)

    E = gw * yw / ((z + 2) * gw * yw + 2) -\
        (gw * yw * (2 * z + 5) + 3) / ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))

    F = (1 / 3 * (z + 2) * (z + 3) * gw * yw**2 + 2 * (z + 2) * yw) / ((z + 2) * gw * yw + 2) -\
        1 / 2 * (z + 2) * (z + 3) * (z + 4) * (gw * yw**2 + 3 * (z + 2) * (z + 3) * yw) /\
        ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))

    dyw_dt = (B * F - C * E) / (A * E - D * B)
    dz_dt = (C * D - A * F) / (A * E - D * B)

    return [dyw_dt, dz_dt]


tau = np.arange(0, 1, 0.001)
z0 = 10
yw_z_0 = [1, z0, ]

# yw_z_sol = Boyd_odeint(model_general_cut, yw_z_0, tau)
yw_z_sol = RK45(model_general_cut, t0=0, y0=yw_z_0, t_bound=1)

yw_sol = yw_z_sol[:, 0]
z_sol = yw_z_sol[:, 1]

plt.figure(dpi=300)

# plt.plot(1 - M1w_sol, yw_sol, 'o-')
# plt.plot(1 - M1w_sol, xw_sol, 'o-')
# plt.plot(1 - M1w_sol, z_sol, 'o-')

# plt.plot(tau, yw_sol)
# plt.plot(tau, yw_sol)
plt.plot(tau, z_sol)

# plt.ylim(0, 1)

plt.show()
