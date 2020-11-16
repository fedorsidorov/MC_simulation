import numpy as np
from scipy. integrate import odeint
import matplotlib.pyplot as plt


# %%
def solve_system_most_probable(alpha, z0, tau):

    gw = alpha / (z0 + 1)

    def model_most_probable(yw_z_M1w, tau):

        yw, z, M1w = yw_z_M1w[0], yw_z_M1w[1], yw_z_M1w[2]

        dyw_dt = - yw**2
        dz_dt = 0
        dM1w_dt = -M1w * 2 * yw / (1 + gw * yw)

        return [dyw_dt, dz_dt, dM1w_dt]

    yw_z_M1w_0 = [1, 0, 1]

    yw_z_M1w = odeint(model_most_probable, yw_z_M1w_0, tau)

    yw = yw_z_M1w[:, 0]
    z = yw_z_M1w[:, 1]
    M1w = yw_z_M1w[:, 2]

    xw = yw * (z + 1) / (z0 + 1)

    return xw, yw, z, M1w


def solve_system_general(alpha, z0, tau):

    gw = alpha / (z0 + 1)

    def model(yw_z_M1w, tau):

        yw, z, M1w = yw_z_M1w[0], yw_z_M1w[1], yw_z_M1w[2]

        A = -(1 / (yw * (1 + gw * yw * (z + 1))) + (z + 2) * gw / ((z + 2) * gw * yw + 2))

        B = -(1 / ((z + 1) * (1 + gw * yw * (z + 1))) + gw * yw / ((z + 2) * gw * yw + 2))

        C = yw * (z + 1) / (gw * yw * (z + 1) + 1) - (1 / 3 * (z + 2) * (z + 3) * gw * yw ** 2 + 2 * (z + 2) * yw) / \
            ((z + 2) * gw * yw + 2)

        D = (z + 2) * gw / ((z + 2) * gw * yw + 2) - (2 * (z + 2) * (z + 3) * gw * yw + 3 * (z + 2)) / \
            ((z + 2) * (z + 3) * gw * yw ** 3 + 3 * (z + 2) * yw)

        E = gw * yw / ((z + 2) * gw * yw + 2) - (gw * yw * (2 * z + 5) + 3) / ((z + 2) * (z + 3) * gw * yw +
                                                                               3 * (z + 2))

        F = (1 / 3 * (z + 2) * (z + 3) * gw * yw ** 2 + 2 * (z + 2) * yw) / ((z + 2) * gw * yw + 2) - \
            1 / 2 * (z + 2) * (z + 3) * (z + 4) * (gw * yw ** 2 + 3 * (z + 2) * (z + 3) * yw) / \
            ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))

        dyw_dt = (B * F - C * E) / (A * E - D * B)
        dz_dt = (C * D - A * F) / (A * E - D * B)
        dM1w_dt = M1w * (1 / yw * dyw_dt + 1 / (z + 1) * dz_dt - yw * (z + 1)) / (1 + gw * yw * (z + 1))

        return [dyw_dt, dz_dt, dM1w_dt]

    yw_z_M1w_0 = [1, z0, 1]

    yw_z_M1w = odeint(model, yw_z_M1w_0, tau)

    yw = yw_z_M1w[:, 0]
    z = yw_z_M1w[:, 1]
    M1w = yw_z_M1w[:, 2]

    xw = yw * (z + 1) / (z0 + 1)

    return xw, yw, z, M1w


# %%
tau = np.arange(0, 10, 0.001)

alpha = 0.1
z0 = 10

# xw_sol, yw_sol, z_sol, M1w_sol = solve_system_general(alpha, z0, tau)
xw_sol, yw_sol, z_sol, M1w_sol = solve_system_most_probable(alpha, z0, tau)

plt.figure(dpi=300)
plt.plot(1 - M1w_sol, yw_sol)

# plt.ylim(0, 5)

plt.show()

