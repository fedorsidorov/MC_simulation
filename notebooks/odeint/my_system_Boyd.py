import numpy as np
from scipy. integrate import odeint
import matplotlib.pyplot as plt


def get_A(gw, yw, z):

    return -(1 / (yw * (1 + gw * yw * (z + 1))) + (z + 2) * gw / ((z + 2) * gw * yw + 2))


def get_B(gw, yw, z):
    return -(1 / ((z + 1) * (1 + gw * yw * (z + 1))) + gw * yw / ((z + 2) * gw * yw + 2))


def get_C(gw, yw, z):
    return yw * (z + 1) / (gw * yw * (z + 1) + 1) - \
        (1 / 3 * (z + 2) * (z + 3) * gw * yw ** 2 + 2 * (z + 2) * yw) / \
        ((z + 2) * gw * yw + 2)


def get_D(gw, yw, z):
    return (z + 2) * gw / ((z + 2) * gw * yw + 2) - \
        (2 * (z + 2) * (z + 3) * gw * yw + 3 * (z + 2)) / \
        ((z + 2) * (z + 3) * gw * yw ** 2 + 3 * (z + 2) * yw)


def get_E(gw, yw, z):
    return gw * yw / ((z + 2) * gw * yw + 2) - \
        (gw * yw * (2 * z + 5) + 3) / ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))


def get_F(gw, yw, z):

    num_1 = 1/3 * (z+2) * (z+3) * gw * yw**2 + 2 * (z+2) * yw
    den_1 = (z+2) * gw * yw + 2

    num_2 = 1/2 * (z+2) * (z+3) * (z+4) * (gw * yw**2 + 3 * (z+2) * (z+3) * yw)
    den_2 = (z+2) * (z+3) * gw * yw + 3 * (z+2)

    # print(num_1, den_1, num_2, den_2)

    return num_1/den_1 - num_2/den_2

    # return (1 / 3 * (z + 2) * (z + 3) * gw * yw**2 + 2 * (z + 2) * yw) / ((z + 2) * gw * yw + 2) -\
    #     1 / 2 * (z + 2) * (z + 3) * (z + 4) * (gw * yw**2 + 3 * (z + 2) * (z + 3) * yw) /\
    #     ((z + 2) * (z + 3) * gw * yw + 3 * (z + 2))


def get_dyw_dt(gw, yw, z):
    return (get_B(gw, yw, z) * get_F(gw, yw, z) - get_C(gw, yw, z) * get_E(gw, yw, z)) /\
           (get_A(gw, yw, z) * get_E(gw, yw, z) - get_D(gw, yw, z) * get_B(gw, yw, z))


def get_dz_dt(gw, yw, z):
    return (get_C(gw, yw, z) * get_D(gw, yw, z) - get_A(gw, yw, z) * get_F(gw, yw, z)) /\
           (get_A(gw, yw, z) * get_E(gw, yw, z) - get_D(gw, yw, z) * get_B(gw, yw, z))


def get_dM1w_dt(gw, yw, z, M1w):
    return M1w * (1 / yw * get_dyw_dt(gw, yw, z) + 1 / (z + 1) * get_dz_dt(gw, yw, z) - yw * (z + 1)) /\
           (1 + gw * yw * (z + 1))


def model_general(t, yw_z_M1w):

    z0 = 10
    alpha = 0.1
    gw = alpha / (z0 + 1)

    yw, z, M1w = yw_z_M1w[0], yw_z_M1w[1], yw_z_M1w[2]

    dyw_dt = get_dyw_dt(gw, yw, z)
    dz_dt = get_dz_dt(gw, yw, z)
    dM1w_dt = get_dM1w_dt(gw, yw, z, M1w)

    return np.array([dyw_dt, dz_dt, dM1w_dt])


def RK4_PCH(model, y_0, tt):

    y_i = y_0

    h = tt[1] - tt[0]

    y_sol = np.zeros((len(tt), 3))
    y_sol[0, :] = y_i

    for i, t in enumerate(tt[:-1]):

        y_i = y_sol[i, :]
        print(y_i)

        if i < 3:  # Runge-Kutta

            K1 = model(t, y_i)
            K2 = model(t + h/2, y_i + h/2 * K1)
            K3 = model(t + h/2, y_i + h/2 * K2)
            K4 = model(t + h, y_i + h * K3)

            print('K:', K1, K2, K3, K4)
            print('K sum:', K1 + K2 + K3 + K4)

            y_new = y_i + h/6 * (K1 + 2*K2 + 2*K3 + K4)


            print('y_new = ', y_new)

            y_sol[i+1, :] = y_new


        else:  # Hamming

            f_i = model(tt[i], y_sol[i, :])
            f_i_1 = model(tt[i-1], y_sol[i-1, :])
            f_i_2 = model(tt[i-2], y_sol[i-2, :])

            # Adams
            # xy_pred = xy_sol[i, :] + h/12 * (23*f_i - 16*f_i_1 + 5*f_i_2)
            # xy_new = xy_sol[i, :] + h/24 * (f_i_2 - 5*f_i_1 + 19*f_i + 9*model(tt[i+1], xy_pred))

            # Hamming
            y_pred = y_sol[i - 3, :] + 4*h/3 * (2*f_i - f_i_1 + 2*f_i_2)
            y_new = 1/8 * (9*y_sol[i, :] - y_sol[i-2, :]) + 3/8 * h * (-f_i_1 + 2*f_i + model(tt[i+1], y_pred))

            y_sol[i + 1, :] = y_new

    return y_sol


# %%
yw_0 = 1
z_0 = 2
M1w_0 = 1
gw = 0.1 / (10 + 1)

#  1/2 * (z+2) * (z+3) * (z+4) * (gw * yw**2 + 3 * (z+2) * (z+3) * yw)

A = get_A(gw, yw_0, z_0)
B = get_B(gw, yw_0, z_0)
C = get_C(gw, yw_0, z_0)
D = get_D(gw, yw_0, z_0)
E = get_E(gw, yw_0, z_0)
F = get_F(gw, yw_0, z_0)

print(A, B, C, D, E, F)
print((B*F - C*E) / (A*E - D*B))
print((C*D - A*F) / (A*E - D*B))

yw_z_M1w_0 = np.array([yw_0, z_0, M1w_0])
t_total = 10
t_step = 0.001

# tt = np.arange(0, t_total, t_step)
tt = [0, 1]

solution = RK4_PCH(model_general, yw_z_M1w_0, tt)

# %%
yw = solution[:, 0]
z = solution[:, 1]
M1w = solution[:, 2]

plt.figure(dpi=300)
plt.plot(tt, z)
# plt.plot(tt, y_res)

plt.show()



