import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mapping import mapping_3p3um_80nm as mm
from functions import MC_functions as mcf
import constants as const
import copy
from scipy import special

const = importlib.reload(const)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)


# %%
def get_final_x_arr(x0_arr, D, delta_t):
    N_arr = np.random.normal(size=len(x0_arr))
    x_arr = x0_arr + np.sqrt(D * delta_t) * N_arr * 1e+7  # cm to nm
    return x_arr


def get_final_z_arr(z0_arr_raw, d_PMMA, D, delta_t):

    z0_arr = (d_PMMA - z0_arr_raw) * 1e-7  # nm -> cm
    # z0_arr = z0_arr_raw

    sqrt = np.sqrt(4 * D * delta_t)
    N0_arr = 1/2 * special.erfc(-z0_arr / sqrt)
    U_arr = np.random.random(len(z0_arr))
    V_arr = np.random.random(len(z0_arr))

    arg_if_arr = V_arr * special.erfc(-z0_arr / sqrt)
    arg_else_arr = V_arr * special.erfc(z0_arr / sqrt)

    # z_arr = -z0_arr + sqrt * special.erfcinv(arg_else_arr) * 1e+7  # cm to nm
    z_arr = -z0_arr + sqrt * special.erfcinv(arg_else_arr)

    inds_if = np.where(U_arr < N0_arr)[0]
    # z_arr[inds_if] = z0_arr[inds_if] + sqrt * special.erfcinv(arg_if_arr[inds_if]) * 1e+7  # cm to nm
    z_arr[inds_if] = z0_arr[inds_if] + sqrt * special.erfcinv(arg_if_arr[inds_if])

    z_arr_final = d_PMMA - z_arr * 1e+7
    # z_arr_final = z_arr
    return z_arr_final


def get_D(T_C, wp):  # in cm^2 / s
    Tg = 120
    delta_T = T_C - Tg

    wp_edge = 0.75

    if T_C == 160:
        wp_edge = 0.9473886
    elif T_C == 125:
        wp_edge = 0.875

    if wp < wp_edge:  # 1st region
        C1, C2, C3, C4 = -4.428, 1.842, 0, 8.12e-3
    else:  # 2nd region
        C1, C2, C3, C4 = 26.0, 37.0, 0.0797, 0

    log_D = (C1 - wp * C2) + delta_T * C3 + wp * delta_T * C4

    return 10**log_D


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

        # n_arr[N - 1] = (2 * 15now21 * tau * betas[N - 2] + h ** 2 * n_arr[N - 1]) /\
        #                (2 * 15now21 * tau * (1 - alphas[N - 2]) + h ** 2)

        n_arr[N - 1] = n_end

        for i in range(N-2, -1, -1):
            n_arr[i] = alphas[i] * n_arr[i + 1] + betas[i]

    return n_arr


def make_simple_diffusion_sim(conc_matrix, D, x_len, z_len, time_step, h_nm, total_time):

    now_time = 0

    while now_time < total_time:

        now_time += time_step

        for k in range(z_len):

            conc_matrix[:, k] = get_concentration_1d_arr_bnd2_0(
                n0_arr=conc_matrix[:, k],
                D=D,
                tau=time_step,
                h=h_nm * 1e-7,
                total_time=total_time
            )

            for i in range(x_len):
                conc_matrix[i, :] = get_concentration_1d_arr_bnd1(
                    n0_arr=conc_matrix[i, :],
                    n_beg=0,
                    n_end=conc_matrix[i, -1],
                    D=D,
                    tau=time_step,
                    h=h_nm * 1e-7,
                    total_time=total_time
                )

    return conc_matrix


# %%
# L = 0.1
# lamda = 46
# C = 460
# rho = 7800
# T0 = 20
# T_l = 300
# T_r = 100
# delta_t = 60
#
# 15now21 = lamda / rho / C
#
# t_end = total_time = 10
# # t_end = total_time = 60000
# tau = t_end / 100
#
# N = 100
# h = L / (N-1)
# n0_arr = np.ones(100) * 20
#
# n_final_arr = get_concentration_1d_arr_bnd1(n0_arr, T_l, T_r, 15now21, tau, h, total_time)
# # n_final_arr = get_concentration_1d_arr_bnd2_0(n0_arr, 15now21, tau, h, total_time)
# # n_final_arr = get_concentration_1d_arr_bnd_12_0(n0_arr, T_r, 15now21, tau, h, total_time)
#
# plt.figure(dpi=300)
# plt.plot(n_final_arr)
# plt.xlim(0, 100)
# plt.ylim(0, 300)
# plt.show()
