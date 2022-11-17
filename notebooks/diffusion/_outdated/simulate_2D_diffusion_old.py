import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import importlib

from mapping import mapping_3um_500nm as mm
from functions import diffusion_functions

df = importlib.reload(diffusion_functions)
mm = importlib.reload(mm)


# %%
def get_concentration_1d_arr_bnd1(n0_arr, n_beg, n_end, D, tau, h, total_time):
    N = len(n0_arr)

    n_arr = n0_arr

    alphas = np.zeros(N)
    betas = np.zeros(N)

    now_time = 0

    while now_time < total_time:

        now_time += tau

        alphas[0] = 0
        betas[0] = n_beg

        for i in range(1, N-1):
            Ai = Ci = D / h**2
            Bi = 2 * D / h**2 + 1 / tau
            Fi = -n_arr[i] / tau

            alphas[i] = Ai / (Bi - Ci * alphas[i - 1])
            betas[i] = (Ci * betas[i - 1] - Fi) / (Bi - Ci * alphas[i - 1])

        n_arr[N - 1] = n_end

        for i in range(N-2, -1, -1):
            n_arr[i] = alphas[i] * n_arr[i + 1] + betas[i]

    return n_arr


def get_concentration_1d_arr_bnd2_0(n0_arr, D, tau, h, total_time):
    N = len(n0_arr)

    n_arr = n0_arr

    alphas = np.zeros(N)
    betas = np.zeros(N)

    now_time = 0

    while now_time < total_time:

        now_time += tau

        alphas[0] = 2 * D * tau / (h**2 + 2 * D * tau)
        betas[0] = h**2 * n_arr[0] / (h**2 + 2 * D * tau)

        for i in range(1, N-1):
            Ai = Ci = D / h**2
            Bi = 2 * D / h**2 + 1 / tau
            Fi = -n_arr[i] / tau

            alphas[i] = Ai / (Bi - Ci * alphas[i - 1])
            betas[i] = (Ci * betas[i - 1] - Fi) / (Bi - Ci * alphas[i - 1])

        n_arr[N - 1] = (2 * D * tau * betas[N - 2] + h ** 2 * n_arr[N - 1]) /\
                       (2 * D * tau * (1 - alphas[N - 2]) + h ** 2)

        for i in range(N-2, -1, -1):
            n_arr[i] = alphas[i] * n_arr[i + 1] + betas[i]

    return n_arr


def get_concentration_1d_arr_bnd_12_0(n0_arr, n_end, D, tau, h, total_time):
    N = len(n0_arr)

    n_arr = n0_arr

    alphas = np.zeros(N)
    betas = np.zeros(N)

    now_time = 0

    while now_time < total_time:

        now_time += tau

        alphas[0] = 2 * D * tau / (h**2 + 2 * D * tau)
        betas[0] = h**2 * n_arr[0] / (h**2 + 2 * D * tau)

        for i in range(1, N-1):
            Ai = Ci = D / h**2
            Bi = 2 * D / h**2 + 1 / tau
            Fi = -n_arr[i] / tau

            alphas[i] = Ai / (Bi - Ci * alphas[i - 1])
            betas[i] = (Ci * betas[i - 1] - Fi) / (Bi - Ci * alphas[i - 1])

        n_arr[N - 1] = (2 * D * tau * betas[N - 2] + h ** 2 * n_arr[N - 1]) /\
                       (2 * D * tau * (1 - alphas[N - 2]) + h ** 2)

        # n_arr[N - 1] = n_end

        for i in range(N-2, -1, -1):
            n_arr[i] = alphas[i] * n_arr[i + 1] + betas[i]

    return n_arr


def get_concentration_1d_arr_bnd_12_0_NEW(n0_arr, D, tau, h, total_time):
    N = len(n0_arr)

    n_arr = n0_arr

    alphas = np.zeros(N)
    betas = np.zeros(N)

    now_time = 0

    while now_time < total_time:

        now_time += tau

        # alphas[0] = 2 * D * tau / (h**2 + 2 * D * tau)
        # betas[0] = h**2 * n_arr[0] / (h**2 + 2 * D * tau)

        alphas[0] = 0
        betas[0] = 0

        for i in range(1, N-1):
            Ai = Ci = D / h**2
            Bi = 2 * D / h**2 + 1 / tau
            Fi = -n_arr[i] / tau

            alphas[i] = Ai / (Bi - Ci * alphas[i - 1])
            betas[i] = (Ci * betas[i - 1] - Fi) / (Bi - Ci * alphas[i - 1])

        n_arr[N - 1] = (2 * D * tau * betas[N - 2] + h ** 2 * n_arr[N - 1]) /\
                       (2 * D * tau * (1 - alphas[N - 2]) + h ** 2)

        # n_arr[N - 1] = n_end

        for i in range(N-2, -1, -1):
            n_arr[i] = alphas[i] * n_arr[i + 1] + betas[i]

    return n_arr


def make_simple_diffusion_sim(conc_matrix, D, x_len, z_len, time_step, h_nm, total_time):

    now_time = 0

    while now_time < total_time:

        now_time += time_step

        for k in range(z_len):

            conc_matrix[:, k] = get_concentration_1d_arr_bnd_12_0_NEW(
                n0_arr=conc_matrix[:, k],
                D=D,
                tau=time_step,
                h=h_nm * 1e-7,
                total_time=total_time
            )

            for i in range(x_len):
                conc_matrix[i, :] = get_concentration_1d_arr_bnd1(
                    n0_arr=conc_matrix[i, :],
                    # n_beg=0,
                    # n_end=0,
                    n_beg=conc_matrix[i, 0],
                    n_end=conc_matrix[i, -1],
                    D=D,
                    tau=time_step,
                    h=h_nm * 1e-7,
                    total_time=total_time
                )

    return conc_matrix


# %%
scission_matrix = np.load('notebooks/DEBER_simulation/test_scission_matrix.npy')

zip_length = 1000
monomer_matrix = scission_matrix * zip_length

conc_matrix_0 = monomer_matrix / (mm.step_50nm * mm.ly * mm.step_50nm * 1e-21)

# plt.figure(dpi=300)
# plt.imshow(conc_matrix_0.transpose())
# plt.title('monomer concentration, cm$^{-3}$')
# plt.colorbar()
# plt.show()

conc_matrix = deepcopy(conc_matrix_0)
# D = df.get_D(130, 1)  # 2.4e-9
D = 1e-9

xx = mm.x_centers_50nm * 1e-7
zz = mm.z_centers_50nm * 1e-7

time_step = 0.1
total_time = 1

conc_matrix = make_simple_diffusion_sim(
    conc_matrix=conc_matrix,
    D=D,
    x_len=len(xx),
    z_len=len(zz),
    time_step=time_step,
    h_nm=mm.step_50nm,
    total_time=total_time
)

plt.figure(dpi=300)
plt.imshow(conc_matrix.transpose())
plt.title('monomer concentration, cm$^{-3}$')
plt.colorbar()
plt.show()

# %%
scission_matrix = np.load('notebooks/DEBER_simulation/test_scission_matrix.npy')

zip_length = 150
monomer_matrix = scission_matrix * zip_length

CC_0 = monomer_matrix / (mm.step_50nm * mm.ly * mm.step_50nm * 1e-21)

# plt.figure(dpi=300)
# plt.imshow(CC_0.transpose())
# plt.title('monomer concentration, cm$^{-3}$')
# plt.colorbar()
# plt.show()

# D = df.get_D(130, 1)  # 2.4e-9
D = 1e-9

xx = mm.x_centers_50nm * 1e-7
yy = mm.z_centers_50nm * 1e-7
h = mm.step_50nm * 1e-7

CC = deepcopy(CC_0)

Nx = len(xx)
Ny = len(zz)

t_end = 0.01
tau = t_end / 10

t = 0

alphas_x = np.zeros(Nx)
betas_x = np.zeros(Nx)

alphas_y = np.zeros(Ny)
betas_y = np.zeros(Ny)


while t < t_end:
    t += tau

    # Ox
    for k in range(0, Ny):
        alphas_x[0] = 0
        # betas_z[0] = T_h
        betas_x[0] = 0

        for i in range(1, Nx - 1):
            Ai = Ci = D / h**2
            Bi = 2 * D / h**2 + 1 / tau
            Fi = -CC[i, k] / tau

            alphas_x[i] = Ai / (Bi - Ci * alphas_x[i - 1])
            betas_x[i] = (Ci * betas_x[i - 1] - Fi) / (Bi - Ci * alphas_x[i - 1])

        # CC[Nx - 1, k] = T_c
        CC[Nx - 1, k] = 0

        for i in range(Nx - 2, -1, -1):
            CC[i, k] = alphas_x[i] * CC[i + 1, k] + betas_x[i]

    # Oy
    for i in range(1, Nx - 1):
        alphas_y[0] = 2.0 * D * tau / (2.0 * D * tau + h**2)
        betas_y[0] = h**2 * CC[i, 0] / (2.0 * D * tau + h**2)

        for j in range(1, Ny - 1):
            Ai = Ci = D / h ** 2
            Bi = 2 * D / h ** 2 + 1 / tau
            Fi = fi = -CC[i, j] / tau
            alphas_y[j] = Ai / (Bi - Ci * alphas_y[j - 1])
            betas_y[j] = (Ci * betas_y[j - 1] - Fi) / (Bi - Ci * alphas_y[j - 1])

        CC[i, Ny - 1] = (2.0 * D * tau * betas_y[Ny - 2] + h**2 * CC[i, Ny - 1]) /\
                        (2.0 * D * tau * (1.0 - alphas_y[Ny - 2]) + h**2)

        for j in range(Ny - 2, -1, -1):
            CC[i, j] = alphas_y[j] * CC[i, j + 1] + betas_y[j]


plt.figure(dpi=300)
plt.imshow(CC.transpose())
plt.title('monomer concentration, cm$^{-3}$')
plt.colorbar()
plt.show()
