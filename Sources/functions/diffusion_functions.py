import importlib
import matplotlib.pyplot as plt
from functions import MC_functions as mcf
import numpy as np
from tqdm import tqdm
import mapping_exp_100_3 as mapping

mcf = importlib.reload(mcf)
mapping = importlib.reload(mapping)


# %%
def get_D(T, wp=1):  # in cm^2 / s
    dT = T - 120
    # wp = 1  # polymer weight fraction
    C1, C2, C3, C4 = 26.0, 37.0, 0.0797, 0
    log_D = wp * dT * C4 + (C1 - wp * C2) + dT * C3
    return 10**log_D


def get_delta_coord(T_C, wp=1):  # nanometers!!!
    D = get_D(T_C, wp)  # in cm^2 / s
    dt = 1e-6  # s
    sigma = np.sqrt(2 * D * dt)
    xx = np.linspace(-3 * sigma, 3 * sigma, 100)
    probs = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-xx ** 2 / (2 * sigma ** 2))
    probs_norm = probs / np.sum(probs)
    return np.random.choice(xx, p=probs_norm) * 1e+7


def get_delta_xyz(D, t):
    x = get_delta_coord(D, t)
    y = get_delta_coord(D, t)
    z = get_delta_coord(D, t)
    return np.array((x, y, z), dtype=float)


def track_monomer(x0, z0, xx, zz_vac, d_PMMA, T_C):  # nanometers!!!

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

    while now_z > get_z_vac_for_x(now_x):  # and pos < pos_max:

        now_x += get_delta_coord(T_C)
        delta_z = get_delta_coord(T_C)

        if now_z + delta_z > d_PMMA:
            now_z -= delta_z
        else:
            now_z += delta_z

        history_x[pos] = now_x
        history_z[pos] = now_z

        pos += 1

        if pos == pos_max:
            print('overload')
            break

    return history_x, history_z, pos


# %%
T_C = 140
delta_t = 1e-6  # s
d_PMMA = 100
x0, z0 = 100, 90

x_arr = mapping.x_centers_2nm
z_vac_arr = np.ones(len(x_arr)) * np.cos(x_arr * np.pi / 2000 / 20) * d_PMMA / 2

N = 1

progress_bar = tqdm(total=N, position=0)

plt.figure(dpi=300)

for i in range(N):
    x_h, z_h, cnt = track_monomer(x0, z0, x_arr, z_vac_arr, d_PMMA, T_C)
    plt.plot(x_h[:cnt], z_h[:cnt], 'o-')
    # z_escape = track_monomer(x0, z0, xx, zz_vac, d_PMMA, T_C)
    progress_bar.update()

plt.plot(x_arr, z_vac_arr)
plt.show()
