import matplotlib.pyplot as plt
import numpy as np


# %% functins
def get_dM1_dt(gw, M1w, yw, z):

    derivative = -(
            9*gw**2*yw**2*z**3 + 41*gw**2*yw**2*z**2 + 58*gw**2*yw**2*z + 24*gw**2*yw**2 + 24*gw*yw*z**2 +
            96*gw*yw*z + 96*gw*yw + 36*z + 72
    ) * M1w*yw / (
            6*gw**3*yw**3*z**3 + 24*gw**3*yw**3*z**2 + 30*gw**3*yw**3*z + 12*gw**3*yw**3 + 18*gw**2*yw**2*z**2 +
            66*gw**2*yw**2*z + 60*gw**2*yw**2 + 36*gw*yw*z + 84*gw*yw + 36
    )

    # derivative = -(
    #         9 * gw ** 2 * yw ** 2 * z ** 3 + 41 * gw ** 2 * yw ** 2 * z ** 2 + 58 * gw ** 2 * yw ** 2 * z +
    #         24 * gw ** 2 * yw ** 2 + 14 * gw * yw * z ** 2 + 58 * gw * yw * z + 60 * gw * yw + 6 * z + 12
    # ) * M1w * yw / (
    #         6 * gw ** 3 * yw ** 3 * z ** 3 + 24 * gw ** 3 * yw ** 3 * z ** 2 + 30 * gw ** 3 * yw ** 3 * z +
    #         12 * gw ** 3 * yw ** 3 + 18 * gw ** 2 * yw ** 2 * z ** 2 + 66 * gw ** 2 * yw ** 2 * z +
    #         60 * gw ** 2 * yw ** 2 + 18 * gw * yw * z + 42 * gw * yw + 6
    # )
    return derivative


def get_dy_dt(gw, M1w, yw, z):

    derivative = (
            gw**3*yw**3*z**5 + 5*gw**3*yw**3*z**4 + 3*gw**3*yw**3*z**3 - 17*gw**3*yw**3*z**2 - 28*gw**3*yw**3*z -
            12*gw**3*yw**3 + 6*gw**2*yw**2*z**4 + 27*gw**2*yw**2*z**3 + 19*gw**2*yw**2*z**2 - 52*gw**2*yw**2*z -
            60*gw**2*yw**2 + 18*gw*yw*z**3 + 42*gw*yw*z**2 - 24*gw*yw*z - 84*gw*yw - 36
            ) * yw**2 / (
            6*(
                gw**3*yw**3*z**3 + 4*gw**3*yw**3*z**2 + 5*gw**3*yw**3*z + 2*gw**3*yw**3 + 3*gw**2*yw**2*z**2 +
                11*gw**2*yw**2*z + 10*gw**2*yw**2 + 6*gw*yw*z + 14*gw*yw + 6
            )
    )

    # derivative = (
    #         gw ** 3 * yw ** 3 * z ** 5 + 5 * gw ** 3 * yw ** 3 * z ** 4 + 3 * gw ** 3 * yw ** 3 * z ** 3 -
    #         17 * gw ** 3 * yw ** 3 * z ** 2 - 28 * gw ** 3 * yw ** 3 * z - 12 * gw ** 3 * yw ** 3 +
    #         2 * gw ** 2 * yw ** 2 * z ** 4 + 5 * gw ** 2 * yw ** 2 * z ** 3 - 25 * gw ** 2 * yw ** 2 * z ** 2 -
    #         90 * gw ** 2 * yw ** 2 * z - 72 * gw ** 2 * yw ** 2 + gw * yw * z ** 3 - gw * yw * z ** 2 -
    #         18 * gw * yw * z - 30 * gw * yw - 6
    #              ) * yw ** 2 / (
    #         6 * (
    #         gw ** 3 * yw ** 3 * z ** 3 + 4 * gw ** 3 * yw ** 3 * z ** 2 + 5 * gw ** 3 * yw ** 3 * z +
    #         2 * gw ** 3 * yw ** 3 + 3 * gw ** 2 * yw ** 2 * z ** 2 + 11 * gw ** 2 * yw ** 2 * z +
    #         10 * gw ** 2 * yw ** 2 + 3 * gw * yw * z + 7 * gw * yw + 1)
    # )
    return derivative


def get_dz_dt(gw, M1w, yw, z):

    derivative = -gw * (
            gw**2*yw**2*z**5 + 9*gw**2*yw**2*z**4 + 31*gw**2*yw**2*z**3 + 51*gw**2*yw**2*z**2 + 40*gw**2*yw**2*z +
            12*gw**2*yw**2 + 6*gw*yw*z**4 + 48*gw*yw*z**3 + 138*gw*yw*z**2 + 168*gw*yw*z + 72*gw*yw + 18*z**3 +
            84*z**2 + 126*z + 60
    ) * yw**2 * z / (
            6*gw**3*yw**3*z**3 + 24*gw**3*yw**3*z**2 + 30*gw**3*yw**3*z + 12*gw**3*yw**3 + 18*gw**2*yw**2*z**2 +
            66*gw**2*yw**2*z + 60*gw**2*yw**2 + 36*gw*yw*z + 84*gw*yw + 36
    )

    # derivative = -gw * (
    #         gw ** 2 * yw ** 2 * z ** 6 + 9 * gw ** 2 * yw ** 2 * z ** 5 + 31 * gw ** 2 * yw ** 2 * z ** 4 +
    #         51 * gw ** 2 * yw ** 2 * z ** 3 + 40 * gw ** 2 * yw ** 2 * z ** 2 + 12 * gw ** 2 * yw ** 2 * z +
    #         2 * gw * yw * z ** 5 + 12 * gw * yw * z ** 4 + 14 * gw * yw * z ** 3 - 36 * gw * yw * z ** 2 -
    #         88 * gw * yw * z - 48 * gw * yw + z ** 4 + 2 * z ** 3 - z ** 2 - 2 * z
    # ) * yw ** 2 / (
    #         6 * gw ** 3 * yw ** 3 * z ** 3 + 24 * gw ** 3 * yw ** 3 * z ** 2 + 30 * gw ** 3 * yw ** 3 * z +
    #         12 * gw ** 3 * yw ** 3 + 18 * gw ** 2 * yw ** 2 * z ** 2 + 66 * gw ** 2 * yw ** 2 * z +
    #         60 * gw ** 2 * yw ** 2 + 18 * gw * yw * z + 42 * gw * yw + 6
    # )
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
tau_total = 20
tau_step = 0.001
tau = np.arange(0, tau_total, tau_step)

yw_0 = 1
z_0 = 10
M1w_0 = 1

solution_0p1_m0p8 = RK4_PCH(model_general, 0.1 / (-0.8 + 1), np.array([M1w_0, yw_0, -0.8]), tau)
solution_0p1_0 = RK4_PCH(model_general, 0.1 / (0 + 1), np.array([M1w_0, yw_0, 0]), tau)
solution_0p1_10 = RK4_PCH(model_general, 0.1 / (10 + 1), np.array([M1w_0, yw_0, 10]), tau)
x_0p1_m0p8 = solution_0p1_m0p8[:, 1] * (solution_0p1_m0p8[:, 2] + 1)
x_0p1_0 = solution_0p1_0[:, 1] * (solution_0p1_0[:, 2] + 1)
x_0p1_10 = solution_0p1_10[:, 1] * (solution_0p1_10[:, 2] + 1)

solution_1_m0p8 = RK4_PCH(model_general, 1 / (-0.8 + 1), np.array([M1w_0, yw_0, -0.8]), tau)
solution_1_0 = RK4_PCH(model_general, 1 / (0 + 1), np.array([M1w_0, yw_0, 0]), tau)
solution_1_10 = RK4_PCH(model_general, 1 / (10 + 1), np.array([M1w_0, yw_0, 10]), tau)
x_1_m0p8 = solution_1_m0p8[:, 1] * (solution_1_m0p8[:, 2] + 1)
x_1_0 = solution_1_0[:, 1] * (solution_1_0[:, 2] + 1)
x_1_10 = solution_1_10[:, 1] * (solution_1_10[:, 2] + 1)

solution_2_m0p8 = RK4_PCH(model_general, 2 / (-0.8 + 1), np.array([M1w_0, yw_0, -0.8]), tau)
solution_2_0 = RK4_PCH(model_general, 2 / (0 + 1), np.array([M1w_0, yw_0, 0]), tau)
solution_2_10 = RK4_PCH(model_general, 2 / (10 + 1), np.array([M1w_0, yw_0, 10]), tau)
x_2_m0p8 = solution_2_m0p8[:, 1] * (solution_2_m0p8[:, 2] + 1)
x_2_0 = solution_2_0[:, 1] * (solution_2_0[:, 2] + 1)
x_2_10 = solution_2_10[:, 1] * (solution_2_10[:, 2] + 1)

solution_10_m0p8 = RK4_PCH(model_general, 10 / (-0.8 + 1), np.array([M1w_0, yw_0, -0.8]), tau)
solution_10_0 = RK4_PCH(model_general, 10 / (0 + 1), np.array([M1w_0, yw_0, 0]), tau)
solution_10_10 = RK4_PCH(model_general, 10 / (10 + 1), np.array([M1w_0, yw_0, 10]), tau)
x_10_m0p8 = solution_10_m0p8[:, 1] * (solution_10_m0p8[:, 2] + 1)
x_10_0 = solution_10_0[:, 1] * (solution_10_0[:, 2] + 1)
x_10_10 = solution_10_10[:, 1] * (solution_10_10[:, 2] + 1)

# %% Figure 1
gr_1_0p1_m0p8 = np.loadtxt('notebooks/odeint/curves/Boyd_1_0.1_-0.8.txt')
gr_1_0p1_10 = np.loadtxt('notebooks/odeint/curves/Boyd_1_0.1_10.txt')

gr_1_1_m0p8 = np.loadtxt('notebooks/odeint/curves/Boyd_1_1_-0.8.txt')
gr_1_1_10 = np.loadtxt('notebooks/odeint/curves/Boyd_1_1_10.txt')

gr_1_10_m0p8 = np.loadtxt('notebooks/odeint/curves/Boyd_1_10_-0.8.txt')
gr_1_10_10 = np.loadtxt('notebooks/odeint/curves/Boyd_1_10_10.txt')

str_1 = r'$x_0/ \gamma^{-1}_0=$'
str_2 = r'$z_0=$'

plt.figure(dpi=300)
plt.plot(1 - solution_0p1_10[:, 0], solution_0p1_10[:, 2], label=str_1 + '0.1 my')
plt.plot(gr_1_0p1_10[:, 0], gr_1_0p1_10[:, 1], '--', label=str_1 + '0.1 Boyd')

plt.plot(1 - solution_1_10[:, 0], solution_1_10[:, 2], label=str_1 + '1 my')
plt.plot(gr_1_1_10[:, 0], gr_1_1_10[:, 1], '--', label=str_1 + '1 Boyd')

plt.plot(1 - solution_10_10[:, 0], solution_10_10[:, 2], label=str_1 + '10 my')
plt.plot(gr_1_10_10[:, 0], gr_1_10_10[:, 1], '--', label=str_1 + '10 Boyd')

plt.grid()
plt.legend(loc='upper right')
plt.xlabel('1 - $M_1/M_{1_0}$')
plt.ylabel('z')
plt.xlim(0, 1.6)
plt.ylim(0, 11)
plt.show()

# %%
plt.figure(dpi=300)

plt.plot(1 - solution_0p1_m0p8[:, 0], solution_0p1_m0p8[:, 2], label=str_1 + '0.1 my')
plt.plot(gr_1_0p1_m0p8[:, 0], gr_1_0p1_m0p8[:, 1], '--', label=str_1 + '0.1 Boyd')

plt.plot(1 - solution_1_m0p8[:, 0], solution_1_m0p8[:, 2], label=str_1 + '1 my')
plt.plot(gr_1_1_m0p8[:, 0], gr_1_1_m0p8[:, 1], '--', label=str_1 + '1 Boyd')

plt.plot(1 - solution_10_m0p8[:, 0], solution_10_m0p8[:, 2], label=str_1 + '10 my')
plt.plot(gr_1_10_m0p8[:, 0], gr_1_10_m0p8[:, 1], '--', label=str_1 + '10 Boyd')

plt.grid()
plt.legend(loc='upper right')
plt.xlabel('1 - $M_1/M_{1_0}$')
plt.ylabel('z')
plt.xlim(0, 1.6)
plt.ylim(-1, 0.5)
plt.show()

# %% Figure 2
gr_2_0p1_m0p8 = np.loadtxt('notebooks/odeint/curves/Boyd_2_0.1_-0.8.txt')
gr_2_0p1_0 = np.loadtxt('notebooks/odeint/curves/Boyd_2_0.1_0.txt')
gr_2_0p1_10 = np.loadtxt('notebooks/odeint/curves/Boyd_2_0.1_10.txt')

gr_2_2_m0p8 = np.loadtxt('notebooks/odeint/curves/Boyd_2_2_-0.8.txt')
gr_2_2_0 = np.loadtxt('notebooks/odeint/curves/Boyd_2_2_0.txt')
gr_2_2_10 = np.loadtxt('notebooks/odeint/curves/Boyd_2_2_10.txt')

gr_2_10_0 = np.loadtxt('notebooks/odeint/curves/Boyd_2_10_0.txt')
gr_2_10_10 = np.loadtxt('notebooks/odeint/curves/Boyd_2_10_10.txt')

plt.figure(dpi=300)

plt.semilogy(tau * (solution_0p1_m0p8[0, 2] + 1), solution_0p1_m0p8[:, 0], label=str_1 + '0.1, ' + str_2 + '-0.8')
plt.semilogy(gr_2_0p1_m0p8[:, 0], gr_2_0p1_m0p8[:, 1], '--', label=str_1 + '0.1, ' + str_2 + '-0.8 Boyd')

plt.semilogy(tau * (solution_0p1_0[0, 2] + 1), solution_0p1_0[:, 0], label=str_1 + '0.1, ' + str_2 + '0')
plt.semilogy(gr_2_0p1_0[:, 0], gr_2_0p1_0[:, 1], '--', label=str_1 + '0.1, ' + str_2 + '0 Boyd')

plt.semilogy(tau * (solution_0p1_10[0, 2] + 1), solution_0p1_10[:, 0], label=str_1 + '0.1, ' + str_2 + '10')
plt.semilogy(gr_2_0p1_10[:, 0], gr_2_0p1_10[:, 1], '--', label=str_1 + '0.1, ' + str_2 + '10 Boyd')

plt.semilogy(tau * (solution_2_m0p8[0, 2] + 1), solution_2_m0p8[:, 0], label=str_1 + '2, ' + str_2 + '-0.8')
plt.semilogy(gr_2_2_m0p8[:, 0], gr_2_2_m0p8[:, 1], '--', label=str_1 + '2, ' + str_2 + '-0.8 Boyd')

plt.semilogy(tau * (solution_2_0[0, 2] + 1), solution_2_0[:, 0], label=str_1 + '2, ' + str_2 + '0')
plt.semilogy(gr_2_2_0[:, 0], gr_2_2_0[:, 1], '--', label=str_1 + '2, ' + str_2 + '0 Boyd')

plt.semilogy(tau * (solution_2_10[0, 2] + 1), solution_2_10[:, 0], label=str_1 + '2, ' + str_2 + '10')
plt.semilogy(gr_2_2_10[:, 0], gr_2_2_10[:, 1], '--', label=str_1 + '2, ' + str_2 + '10 Boyd')

plt.semilogy(tau * (solution_10_0[0, 2] + 1), solution_10_0[:, 0], label=str_1 + '10, ' + str_2 + '0')
plt.semilogy(gr_2_10_0[:, 0], gr_2_10_0[:, 1], '--', label=str_1 + '10, ' + str_2 + '0 Boyd')

plt.semilogy(tau * (solution_10_10[0, 2] + 1), solution_10_10[:, 0], label=str_1 + '10, ' + str_2 + '10')
plt.semilogy(gr_2_10_10[:, 0], gr_2_10_10[:, 1], '--', label=str_1 + '10, ' + str_2 + '10 Boyd')

plt.grid()
plt.legend(loc='upper right')
plt.xlabel('$x_0 k_s t$')
plt.ylabel('$M_1/M_{1_0}$')
plt.xlim(0, 10)
plt.ylim(0.1, 1)
plt.show()

# %% Figure 3
gr_3_0p1_m0p8 = np.loadtxt('notebooks/odeint/curves/Boyd_3_0.1_-0.8.txt')
gr_3_0p1_0 = np.loadtxt('notebooks/odeint/curves/Boyd_3_0.1_0.txt')
gr_3_0p1_10 = np.loadtxt('notebooks/odeint/curves/Boyd_3_0.1_10.txt')
gr_3_10_0 = np.loadtxt('notebooks/odeint/curves/Boyd_3_10_0.txt')

plt.figure(dpi=300)

plt.plot(1 - solution_0p1_m0p8[:, 0], x_0p1_m0p8 / x_0p1_m0p8[0], label=str_1 + '0.1, ' + str_2 + '-0.8')
plt.plot(gr_3_0p1_m0p8[:, 0], gr_3_0p1_m0p8[:, 1], '--', label=str_1 + '0.1, ' + str_2 + '-0.8 Boyd')

plt.plot(1 - solution_0p1_0[:, 0], x_0p1_0 / x_0p1_0[0], label=str_1 + '0.1, ' + str_2 + '0')
plt.plot(gr_3_0p1_0[:, 0], gr_3_0p1_0[:, 1], '--', label=str_1 + '0.1, ' + str_2 + '0 Boyd')

plt.plot(1 - solution_0p1_10[:, 0], x_0p1_10 / x_0p1_10[0], label=str_1 + '0.1, ' + str_2 + '10')
plt.plot(gr_3_0p1_10[:, 0], gr_3_0p1_10[:, 1], '--', label=str_1 + '0.1, ' + str_2 + '10 Boyd')

plt.plot(1 - solution_10_0[:, 0], x_10_0 / x_10_0[0], label=str_1 + '10, ' + str_2 + '0')
plt.plot(gr_3_10_0[:, 0], gr_3_10_0[:, 1], '--', label=str_1 + '10, ' + str_2 + '0 Boyd')

plt.grid()
plt.legend(loc='upper right')
plt.xlabel('$1 - M_1/M_{1_0}$')
plt.ylabel('$x / x_0$')
plt.xlim(0, 1.6)
plt.ylim(0, 1)
plt.show()



