from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from sympy import erfinv
from tqdm import tqdm

# %%
# D = 3.16e-7 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
D = 3.16e-6 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
# D = 3.16e-5 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s

delta_t = 1e-7  # s
sigma = np.sqrt(2 * D * delta_t)

xx = np.linspace(-3 * sigma, 3 * sigma, 100)
probs = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-xx ** 2 / (2 * sigma ** 2))
probs_norm = probs / np.sum(probs)

# plt.figure(dpi=300)
# plt.plot(xx, probs, 'o')
# plt.show()


# %%
def get_delta_coord_fast():
    return np.random.choice(xx, p=probs_norm)


def get_delta_coord(D, t):
    arg = erfinv(np.random.random()) * np.sqrt(2)
    coord = arg * np.sqrt(2 * D * t)
    return coord * np.random.choice([1, -1])


def get_delta_xyz(D, t):
    x = get_delta_coord(D, t)
    y = get_delta_coord(D, t)
    z = get_delta_coord(D, t)
    return np.array((x, y, z), dtype=float)


def track_monomer(xz_0, l_xz):
    now_x = xz_0[0]
    now_z = xz_0[1]

    pos_max = 1000

    # history_x = np.zeros(pos_max)
    # history_z = np.zeros(pos_max)
    # history_x[0] = now_x
    # history_z[0] = now_z

    pos = 1

    while now_z >= 0 and pos < pos_max:
        now_x += get_delta_coord_fast()

        delta_z = get_delta_coord_fast()

        if now_z + delta_z > l_xz[1]:
            now_z -= delta_z
        else:
            now_z += delta_z

        # history_x[pos] = now_x
        # history_z[pos] = now_z

        pos += 1

        # if pos == pos_max:
        #     print('overload')
        #     break

    # return history_x, history_z, pos
    return now_x


# %%
# D = 3.16e-6 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
# delta_t = 1e-6  # s
#
# x_0, z_0 = -356, 44
# l_x, l_z = 3300, 80
#
# N = 10000
#
# progress_bar = tqdm(total=N, position=0)
#
# for i in range(N):
#     x_h, z_h, cnt = track_monomer([x_0, z_0], [l_x, l_z])
    # z_escape = track_monomer([x_0, z_0], [l_x, l_z])
    # progress_bar.update()


# plt.figure(dpi=300)
# plt.plot(x_h, z_h, 'o-')
# plt.show()

# %%
# array = np.zeros((1000, 3))
#
# now_coords = np.array((0, 0, 0), dtype=float)
#
# progress_bar = tqdm(total=len(array), position=0)
#
# for i in range(len(array)):
#     array[i, :] = now_coords
#     now_coords += get_delta_xyz(3.1e-6, 1e-6)
#     progress_bar.update()
#
# # %%
# plt.figure(dpi=300)
# plt.plot(array[:, 0], array[:, 1])
#
# plt.xlim(-0.00005, 0.00005)
# plt.ylim(-0.00005, 0.00005)
# plt.show()
