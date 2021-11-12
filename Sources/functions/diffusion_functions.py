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
# zz_0 = np.ones(1000000) * 2
#
# zz_f = get_final_z_arr(zz_0, 5, D=1e-11, delta_t=0.001)
#
# z_hist, bins = np.histogram(zz_f, bins=np.linspace(-10, 10, 201))
# centers = (bins[:-1] + bins[1:]) / 2
#
# plt.figure(dpi=300)
# plt.plot(centers, z_hist)
# plt.show()

# %%
# T_C = 160
#
# wpwp = np.linspace(0, 1, 100)
# DD = np.zeros(len(wpwp))
#
# for i in range(len(wpwp)):
#     DD[i] = get_D(T_C, wpwp[i])
#
# plt.figure(dpi=300)
# plt.semilogy(wpwp, DD)
#
# plt.xlabel('polymer mass fraction')
# plt.ylabel('D')
#
# plt.grid()
# plt.show()

