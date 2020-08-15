import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from tqdm import tqdm

import mapping_exp_80nm_Camscan as mapping
from functions import DEBER_functions as deber
from functions import diffusion_functions as df

deber = importlib.reload(deber)
df = importlib.reload(df)
mapping = importlib.reload(mapping)

# %%
D = 3.16e-6 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s

delta_t = 1e-7  # s
sigma = np.sqrt(2 * D * delta_t)

xx_probs = np.linspace(-3 * sigma, 3 * sigma, 100)
probs = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-xx_probs ** 2 / (2 * sigma ** 2))
probs_norm = probs / np.sum(probs)


def get_delta_coord_fast():
    return np.random.choice(xx_probs, p=probs_norm)


def track_monomer(xz_0, x_arr, z_vac_arr, d_PMMA):

    history = np.ones((1000, 2)) * -1

    def get_z_vac_for_x(x):
        if x > np.max(x_arr):
            return z_vac_arr[-1]
        elif x < np.min(x_arr):
            return z_vac_arr[0]
        else:
            return interpolate.interp1d(x_arr, z_vac_arr)(x)

    now_x = xz_0[0]
    now_z = xz_0[1]

    history[0, :] = now_x, now_z

    pos_max = 1000

    pos = 1

    while now_z >= get_z_vac_for_x(now_x) and pos < pos_max:
        now_x += df.get_delta_coord_fast() * 1e-7  # nm -> cm
        delta_z = df.get_delta_coord_fast() * 1e-7  # nm -> cm

        if now_z + delta_z > d_PMMA:
            now_z -= delta_z
        else:
            now_z += delta_z

        history[pos, :] = now_x, now_z

        pos += 1

    return history


# %%
# xx = np.load('xx_diffusion.npy')
xx = mapping.x_centers_2nm * 1e-7
zz_vac_list = np.load('zz_vac_list.npy')
zz_vac = zz_vac_list[30]

d_PMMA = 80e-7
hh_vac = d_PMMA - zz_vac

plt.figure(dpi=300)
plt.plot(xx * 1e+7, hh_vac * 1e+7)

plt.ylim(0, 100)
plt.show()

# %%
e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(xx, zz_vac, 100, r_beam=100e-7)
scission_matrix = deber.get_scission_matrix(e_DATA_PMMA_val)

# %%
# scission_matrix_sum_y = np.sum(scission_matrix, axis=1)
scission_matrix_sum_y = np.sum(np.load('scission_matrix_100.npy'), axis=1)

plt.imshow(scission_matrix_sum_y[700:-700])
plt.show()

# %%
sci_pos_arr = np.array(np.where(scission_matrix_sum_y > 0)).transpose()

_, _ = plt.subplots(dpi=600)

fig = plt.gcf()
fig.set_size_inches(10, 3)

font_size = 14

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.plot(xx*1e+7, hh_vac*1e+7)

for sci_coords in sci_pos_arr[0:-1:10]:

    x_ind, z_ind = sci_coords
    n_scissions = int(scission_matrix_sum_y[x_ind, z_ind])
    xz0 = mapping.x_centers_2nm[x_ind] * 1e-7, mapping.z_centers_2nm[z_ind] * 1e-7

    if xz0[1] > hh_vac[x_ind]:
        continue

    plt.plot(xz0[0]*1e+7, (d_PMMA - xz0[1])*1e+7, 'o')

    history_xz = track_monomer(xz0, xx, zz_vac, d_PMMA)
    inds = np.where(history_xz[:, 0] != -1)[0]
    plt.plot(history_xz[inds, 0]*1e+7, (d_PMMA - history_xz[inds, 1])*1e+7, '--')


ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

# plt.xlim(-1000, 1000)
plt.xlim(-1200, 1200)
plt.ylim(0, 100)

plt.xlabel('$x$, нм', fontsize=font_size)
plt.ylabel('$x$, нм', fontsize=font_size)

plt.grid()
# plt.show()

plt.savefig('diffusion_wide.tiff', dpi=600)
