import importlib
import matplotlib.pyplot as plt
from functions import MC_functions as mcf
import numpy as np
from tqdm import tqdm
# from mapping import mapping_3p3um_80nm as mapping

mcf = importlib.reload(mcf)
# mapping = importlib.reload(mapping)


# %%
def get_D(T, wp=1):  # in cm^2 / s
    dT = T - 120
    # wp = 1  # polymer weight fraction
    C1, C2, C3, C4 = 26.0, 37.0, 0.0797, 0
    log_D = wp * dT * C4 + (C1 - wp * C2) + dT * C3
    return 10**log_D


def get_dt_dx_dz(T_C, wp=1):  # nanometers!!!
    D = get_D(T_C, wp)  # in cm^2 / s
    sigma = 5e-7  # bin size in cm
    dt = sigma**2 / 4 / D
    xx = np.linspace(-3 * sigma, 3 * sigma, 100)
    probs = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-xx ** 2 / (2 * sigma ** 2))
    probs_norm = probs / np.sum(probs)
    return dt, np.random.choice(xx, p=probs_norm) * 1e+7, np.random.choice(xx, p=probs_norm) * 1e+7


def track_monomer(x0, z0, xx, zz_vac, d_PMMA, T_C, wp):  # nanometers!!!

    t_step = 1  # second

    def get_z_vac_for_x(x):
        if x > np.max(xx):
            return zz_vac[-1]
        elif x < np.min(xx):
            return zz_vac[0]
        else:
            return mcf.lin_lin_interp(xx, zz_vac)(x)

    pos_max = 100000
    history_x = np.zeros(pos_max)
    history_z = np.zeros(pos_max)
    history_x[0] = x0
    history_z[0] = z0

    pos = 1

    now_x = x0
    now_z = z0

    total_time = 0

    while total_time < t_step:  # and pos < pos_max:

        dt, dx, dz = get_dt_dx_dz(T_C, wp)
        # print(dt)
        total_time += dt

        now_x += dx
        delta_z = dz

        if now_z + delta_z > d_PMMA:
            now_z -= delta_z
        else:
            now_z += delta_z

        if pos >= pos_max:
            print('overload pos_max')
            break

        history_x[pos] = now_x
        history_z[pos] = now_z

        pos += 1

        if now_z < get_z_vac_for_x(now_x):
            # print('vacuum')
            break

    return history_x, history_z, pos


# %%
# T_C = 140
# d_PMMA = 100
# x0, z0 = 0, 99

# x_arr = mapping.x_centers_2nm
# z_vac_arr = np.ones(len(x_arr)) * np.cos(x_arr * np.pi / 2000 / 20) * d_PMMA / 2
# z_vac_arr = np.zeros(len(x_arr))

# plt.plot(x_arr, z_vac_arr)

# N = 10

# progress_bar = tqdm(total=N, position=0)

# plt.figure(dpi=300)

# for i in range(N):
    # x_h, z_h, cnt = track_monomer(x0, z0, x_arr, z_vac_arr, d_PMMA, T_C, 1)
    # plt.plot(x_h[:cnt], z_h[:cnt], 'o-')
    # progress_bar.update()

# plt.plot(x_arr, z_vac_arr)
# plt.xlim(-50, 50)
# plt.ylim(0, 100)
# plt.show()
