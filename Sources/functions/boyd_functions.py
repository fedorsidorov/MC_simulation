import matplotlib.pyplot as plt
import numpy as np


def get_dM1w_dtau(gw, M1w, yw, z):

    derivative = -(
            9 * gw ** 2 * yw ** 2 * z ** 3 + 41 * gw ** 2 * yw ** 2 * z ** 2 + 58 * gw ** 2 * yw ** 2 * z +
            24 * gw ** 2 * yw ** 2 + 24 * gw * yw * z ** 2 + 96 * gw * yw * z + 96 * gw * yw + 36 * z + 72
    ) * M1w * yw / (
            6 * gw ** 3 * yw ** 3 * z ** 3 + 24 * gw ** 3 * yw ** 3 * z ** 2 + 30 * gw ** 3 * yw ** 3 * z +
            12 * gw ** 3 * yw ** 3 + 18 * gw ** 2 * yw ** 2 * z ** 2 + 66 * gw ** 2 * yw ** 2 * z +
            60 * gw ** 2 * yw ** 2 + 36 * gw * yw * z + 84 * gw * yw + 36
            )

    return derivative


def get_dy_dtau(gw, M1w, yw, z):

    derivative = (
            gw ** 3 * yw ** 3 * z ** 5 + 5 * gw ** 3 * yw ** 3 * z ** 4 + 3 * gw ** 3 * yw ** 3 * z ** 3 -
            17 * gw ** 3 * yw ** 3 * z ** 2 - 28 * gw ** 3 * yw ** 3 * z - 12 * gw ** 3 * yw ** 3 +
            6 * gw ** 2 * yw ** 2 * z ** 4 + 27 * gw ** 2 * yw ** 2 * z ** 3 +
            19 * gw ** 2 * yw ** 2 * z ** 2 - 52 * gw ** 2 * yw ** 2 * z - 60 * gw ** 2 * yw ** 2 +
            18 * gw * yw * z ** 3 + 42 * gw * yw * z ** 2 - 24 * gw * yw * z - 84 * gw * yw - 36
            ) * yw ** 2 / (
                6 * (
                    gw ** 3 * yw ** 3 * z ** 3 + 4 * gw ** 3 * yw ** 3 * z ** 2 + 5 * gw ** 3 * yw ** 3 * z +
                    2 * gw ** 3 * yw ** 3 + 3 * gw ** 2 * yw ** 2 * z ** 2 + 11 * gw ** 2 * yw ** 2 * z +
                    10 * gw ** 2 * yw ** 2 + 6 * gw * yw * z + 14 * gw * yw + 6
                )
            )

    return derivative


def get_dz_dtau(gw, M1w, yw, z):

    derivative = -gw * (
            gw ** 2 * yw ** 2 * z ** 5 + 9 * gw ** 2 * yw ** 2 * z ** 4 + 31 * gw ** 2 * yw ** 2 * z ** 3 +
            51 * gw ** 2 * yw ** 2 * z ** 2 + 40 * gw ** 2 * yw ** 2 * z + 12 * gw ** 2 * yw ** 2 +
            6 * gw * yw * z ** 4 + 48 * gw * yw * z ** 3 + 138 * gw * yw * z ** 2 + 168 * gw * yw * z +
            72 * gw * yw + 18 * z ** 3 + 84 * z ** 2 + 126 * z + 60
    ) * yw ** 2 * z / (
            6 * gw ** 3 * yw ** 3 * z ** 3 + 24 * gw ** 3 * yw ** 3 * z ** 2 + 30 * gw ** 3 * yw ** 3 * z +
            12 * gw ** 3 * yw ** 3 + 18 * gw ** 2 * yw ** 2 * z ** 2 + 66 * gw ** 2 * yw ** 2 * z +
            60 * gw ** 2 * yw ** 2 + 36 * gw * yw * z + 84 * gw * yw + 36
    )

    return derivative


def model(tau, gw, M1w_yw_z):
    M1w, yw, z, = M1w_yw_z[0], M1w_yw_z[1], M1w_yw_z[2]

    dM1w_dtau = get_dM1w_dtau(gw, M1w, yw, z)
    dyw_dtau = get_dy_dtau(gw, M1w, yw, z)
    dz_dtau = get_dz_dtau(gw, M1w, yw, z)

    return np.array([dM1w_dtau, dyw_dtau, dz_dtau])


def RK4_PCH(gw, y_0, tau):
    y_i = y_0

    h = tau[1] - tau[0]

    y_sol = np.zeros((len(tau), 3))
    y_sol[0, :] = y_i

    for i, t in enumerate(tau[:-1]):

        y_i = y_sol[i, :]

        if i < 3:  # Runge-Kutta
            K1 = model(t, gw, y_i)
            K2 = model(t + h / 2, gw, y_i + h / 2 * K1)
            K3 = model(t + h / 2, gw, y_i + h / 2 * K2)
            K4 = model(t + h, gw, y_i + h * K3)

            y_new = y_i + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
            y_sol[i + 1, :] = y_new

        else:  # predictor-corrector
            f_i = model(tau[i], gw, y_sol[i, :])
            f_i_1 = model(tau[i - 1], gw, y_sol[i - 1, :])
            f_i_2 = model(tau[i - 2], gw, y_sol[i - 2, :])

            # Hamming
            y_pred = y_sol[i - 3, :] + 4 * h / 3 * (2 * f_i - f_i_1 + 2 * f_i_2)
            y_new = 1 / 8 * (9 * y_sol[i, :] - y_sol[i - 2, :]) + 3 / 8 * h * (
                    -f_i_1 + 2 * f_i + model(tau[i + 1], gw, y_pred))

            y_sol[i + 1, :] = y_new

    return y_sol


def get_zip_len_term_trans(T_C):

    def func(TT, A, k):
        return A * np.exp(k / TT)

    popt_k_d = np.array([1.31544076e+13, -9.17590274e+03])
    popt_k_t = np.array([1209.47832698, -2345.74312909])
    popt_k_f = np.array([938.53242381, -2314.17898843])

    zip_len_term = func(T_C + 273, *popt_k_d) / func(T_C + 273, *popt_k_t)
    zip_len_trans = func(T_C + 273, *popt_k_d) / func(T_C + 273, *popt_k_f)

    return int(zip_len_trans), int(zip_len_term)
