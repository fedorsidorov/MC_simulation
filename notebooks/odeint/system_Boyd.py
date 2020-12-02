import matplotlib.pyplot as plt
import numpy as np


def get_dM1_dt(gw, M1w, yw, z):
    derivative = -(
            9 * gw ** 2 * yw ** 2 * z ** 3 + 41 * gw ** 2 * yw ** 2 * z ** 2 + 58 * gw ** 2 * yw ** 2 * z +
            24 * gw ** 2 * yw ** 2 + 14 * gw * yw * z ** 2 + 58 * gw * yw * z + 60 * gw * yw + 6 * z + 12
    ) * M1w * yw / (
            6 * gw ** 3 * yw ** 3 * z ** 3 + 24 * gw ** 3 * yw ** 3 * z ** 2 + 30 * gw ** 3 * yw ** 3 * z +
            12 * gw ** 3 * yw ** 3 + 18 * gw ** 2 * yw ** 2 * z ** 2 + 66 * gw ** 2 * yw ** 2 * z +
            60 * gw ** 2 * yw ** 2 + 18 * gw * yw * z + 42 * gw * yw + 6
    )
    return derivative


def get_dy_dt(gw, M1w, yw, z):
    derivative = (
            gw ** 3 * yw ** 3 * z ** 5 + 5 * gw ** 3 * yw ** 3 * z ** 4 + 3 * gw ** 3 * yw ** 3 * z ** 3 -
            17 * gw ** 3 * yw ** 3 * z ** 2 - 28 * gw ** 3 * yw ** 3 * z - 12 * gw ** 3 * yw ** 3 +
            2 * gw ** 2 * yw ** 2 * z ** 4 + 5 * gw ** 2 * yw ** 2 * z ** 3 - 25 * gw ** 2 * yw ** 2 * z ** 2 -
            90 * gw ** 2 * yw ** 2 * z - 72 * gw ** 2 * yw ** 2 + gw * yw * z ** 3 - gw * yw * z ** 2 -
            18 * gw * yw * z - 30 * gw * yw - 6
                 ) * yw ** 2 / (
            6 * (
            gw ** 3 * yw ** 3 * z ** 3 + 4 * gw ** 3 * yw ** 3 * z ** 2 + 5 * gw ** 3 * yw ** 3 * z +
            2 * gw ** 3 * yw ** 3 + 3 * gw ** 2 * yw ** 2 * z ** 2 + 11 * gw ** 2 * yw ** 2 * z +
            10 * gw ** 2 * yw ** 2 + 3 * gw * yw * z + 7 * gw * yw + 1)
    )
    return derivative


def get_dz_dt(gw, M1w, yw, z):
    derivative = -gw * (
            gw ** 2 * yw ** 2 * z ** 6 + 9 * gw ** 2 * yw ** 2 * z ** 5 + 31 * gw ** 2 * yw ** 2 * z ** 4 +
            51 * gw ** 2 * yw ** 2 * z ** 3 + 40 * gw ** 2 * yw ** 2 * z ** 2 + 12 * gw ** 2 * yw ** 2 * z +
            2 * gw * yw * z ** 5 + 12 * gw * yw * z ** 4 + 14 * gw * yw * z ** 3 - 36 * gw * yw * z ** 2 -
            88 * gw * yw * z - 48 * gw * yw + z ** 4 + 2 * z ** 3 - z ** 2 - 2 * z
    ) * yw ** 2 / (
            6 * gw ** 3 * yw ** 3 * z ** 3 + 24 * gw ** 3 * yw ** 3 * z ** 2 + 30 * gw ** 3 * yw ** 3 * z +
            12 * gw ** 3 * yw ** 3 + 18 * gw ** 2 * yw ** 2 * z ** 2 + 66 * gw ** 2 * yw ** 2 * z +
            60 * gw ** 2 * yw ** 2 + 18 * gw * yw * z + 42 * gw * yw + 6
    )
    return derivative


def model_general(t, gw, M1w_yw_z):
    # z0 = 10
    # alpha = 0.1
    # gw = alpha / (z0 + 1)

    M1w, yw, z, = M1w_yw_z[0], M1w_yw_z[1], M1w_yw_z[2]

    dM1w_dt = get_dM1_dt(gw, M1w, yw, z)
    dyw_dt = get_dy_dt(gw, M1w, yw, z)
    dz_dt = get_dz_dt(gw, M1w, yw, z)

    return np.array([dM1w_dt, dyw_dt, dz_dt])


def RK4_PCH(model, gw, y_0, tt):
    y_i = y_0

    h = tt[1] - tt[0]

    y_sol = np.zeros((len(tt), 3))
    y_sol[0, :] = y_i

    for i, t in enumerate(tt[:-1]):

        y_i = y_sol[i, :]
        # print(y_i)

        if i < 3:  # Runge-Kutta

            K1 = model(t, gw, y_i)
            K2 = model(t + h / 2, gw, y_i + h / 2 * K1)
            K3 = model(t + h / 2, gw, y_i + h / 2 * K2)
            K4 = model(t + h, gw, y_i + h * K3)

            # print('K:', K1, K2, K3, K4)
            # print('K sum:', K1 + K2 + K3 + K4)

            y_new = y_i + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

            # print('y_new = ', y_new)

            y_sol[i + 1, :] = y_new

        else:  # predictor-corrector
            f_i = model(tt[i], gw, y_sol[i, :])
            f_i_1 = model(tt[i - 1], gw, y_sol[i - 1, :])
            f_i_2 = model(tt[i - 2], gw, y_sol[i - 2, :])

            # Adams
            # xy_pred = xy_sol[i, :] + h/12 * (23*f_i - 16*f_i_1 + 5*f_i_2)
            # xy_new = xy_sol[i, :] + h/24 * (f_i_2 - 5*f_i_1 + 19*f_i + 9*model(tt[i+1], xy_pred))

            # Hamming
            y_pred = y_sol[i - 3, :] + 4 * h / 3 * (2 * f_i - f_i_1 + 2 * f_i_2)
            y_new = 1 / 8 * (9 * y_sol[i, :] - y_sol[i - 2, :]) + 3 / 8 * h * (
                    -f_i_1 + 2 * f_i + model(tt[i + 1], gw, y_pred))

            y_sol[i + 1, :] = y_new

    return y_sol


# %%
yw_0 = 1
z_0 = 10
# z_0 = -0.8
M1w_0 = 1

# alpha = 0.1
# alpha = 1
alpha = 10

gw = alpha / (z_0 + 1)

M1w_yw_z_0 = np.array([M1w_0, yw_0, z_0])
t_total = 20
t_step = 0.001

tt = np.arange(0, t_total, t_step)
# tt = [0, 1]

solution = RK4_PCH(model_general, gw, M1w_yw_z_0, tt)

# %
M1w = solution[:, 0]
yw = solution[:, 1]
z = solution[:, 2]

plt.figure(dpi=300)
plt.plot(1 - M1w, z)
# plt.plot(tt, y_res)

plt.xlim(0, 1)
plt.ylim(-1, 10)

plt.show()
