import numpy as np
from scipy. integrate import odeint
import matplotlib.pyplot as plt


def model(t, xy):
    x = xy[0]
    y = xy[1]

    dxdt = y
    dydt = -x - y

    return np.array((dxdt, dydt))


def RK4_PCH(model, xy_0, tt):

    xy_i = xy_0

    h = tt[1] - tt[0]

    xy_sol = np.zeros((len(tt), 2))
    xy_sol[0, :] = xy_i

    for i, t in enumerate(tt[:-1]):

        xy_i = xy_sol[i, :]

        if i < 3:  # Runge-Kutta

            K1 = model(t, xy_i)
            K2 = model(t + h/2, xy_i + h/2 * K1)
            K3 = model(t + h/2, xy_i + h/2 * K2)
            K4 = model(t + h, xy_i + h * K3)

            xy_new = xy_i + h/6 * (K1 + 2*K2 + 2*K3 + K4)
            xy_sol[i+1, :] = xy_new

        else:  # Hamming

            f_i = model(tt[i], xy_sol[i, :])
            f_i_1 = model(tt[i-1], xy_sol[i-1, :])
            f_i_2 = model(tt[i-2], xy_sol[i-2, :])

            # Adams
            # xy_pred = xy_sol[i, :] + h/12 * (23*f_i - 16*f_i_1 + 5*f_i_2)
            # xy_new = xy_sol[i, :] + h/24 * (f_i_2 - 5*f_i_1 + 19*f_i + 9*model(tt[i+1], xy_pred))

            # Hamming
            xy_pred = xy_sol[i - 3, :] + 4*h/3 * (2*f_i - f_i_1 + 2*f_i_2)
            xy_new = 1/8 * (9*xy_sol[i, :] - xy_sol[i-2, :]) + 3/8 * h * (-f_i_1 + 2*f_i + model(tt[i+1], xy_pred))

            xy_sol[i + 1, :] = xy_new

    return xy_sol


xy_0 = np.array([1, 0])
t_total = 10
t_step = 0.001
tt = np.arange(0, t_total, t_step)

xy_res = RK4_PCH(model, xy_0, tt)

# %
x_res = xy_res[:, 0]
y_res = xy_res[:, 1]

plt.figure(dpi=300)
plt.plot(tt, x_res)
plt.plot(tt, y_res)

plt.show()



