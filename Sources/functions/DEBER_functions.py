import copy
import importlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from tqdm import tqdm

import MC_classes_DEBER as mcd
import constants as const
import indexes as ind
import mapping_exp_80nm_Camscan as mapping
from functions import array_functions as af
from functions import diffusion_functions as df
from functions import e_matrix_functions as emf
from functions import reflow_functions as rf
from functions import scission_functions as sf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcd = importlib.reload(mcd)
af = importlib.reload(af)
df = importlib.reload(df)
rf = importlib.reload(rf)
sf = importlib.reload(sf)


# %%
def get_e_DATA_PMMA_val(xx, zz_vac, n_electrons, r_beam=100e-7):
    d_PMMA = 80e-7
    ly = mapping.l_y * 1e-7
    # r_beam = 100e-7

    E0 = 20e+3

    structure = mcd.Structure(
        d_PMMA=d_PMMA,
        xx=xx,
        zz_vac=zz_vac,
        ly=ly)

    simulator = mcd.Simulator(
        structure=structure,
        n_electrons=n_electrons,
        E0_eV=E0,
        r_beam=r_beam
    )
    simulator.prepare_e_deque()
    simulator.start_simulation()

    e_DATA = simulator.get_total_history()
    e_DATA_PMMA_val = e_DATA[np.where(np.logical_and(
        e_DATA[:, ind.DATA_layer_id_ind] == ind.PMMA_ind,
        e_DATA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)
    )]

    return e_DATA_PMMA_val


def get_scission_matrix(e_DATA_PMMA_val):
    deg_paths = sf.degpaths_all_WO_Oval
    weight = 0.35

    af.snake_array(
        array=e_DATA_PMMA_val,
        x_ind=ind.DATA_x_ind,
        y_ind=ind.DATA_y_ind,
        z_ind=ind.DATA_z_ind,
        xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
        xyz_max=[mapping.x_max, mapping.y_max, np.inf]
    )

    scissions = sf.get_scissions(e_DATA_PMMA_val, deg_paths, weight=weight)

    scission_matrix = np.histogramdd(
        sample=e_DATA_PMMA_val[:, ind.DATA_coord_inds],
        bins=mapping.bins_2nm,
        weights=scissions
    )[0]

    return scission_matrix


def track_monomer(xz_0, xx, zz_vac, d_PMMA):
    def get_z_vac_for_x(x):
        # return 0
        if x > np.max(xx):
            return zz_vac[-1]
        elif x < np.min(xx):
            return zz_vac[0]
        else:
            return interpolate.interp1d(xx, zz_vac)(x)

    now_x = xz_0[0]
    now_z = xz_0[1]

    pos_max = 1000

    pos = 1

    now_z_vac = get_z_vac_for_x(now_x)

    while now_z >= now_z_vac and pos < pos_max:
        now_x += df.get_delta_coord_fast() * 1e-7  # cm -> nm
        delta_z = df.get_delta_coord_fast() * 1e-7  # cm -> nm

        if now_z + delta_z > d_PMMA:
            now_z -= delta_z
        else:
            now_z += delta_z

        pos += 1

    return now_x


def get_profile_after_diffusion(scission_matrix, zip_length, xx, zz_vac, d_PMMA, double):

    # D = 3.16e-6 * 1e+7 ** 2  # cm^2 / s -> nm^2 / s
    # delta_t = 1e-7  # s

    scission_matrix_sum_y = np.sum(scission_matrix, axis=1)
    n_monomers_groups = zip_length // 10
    x_escape_array = np.zeros(int(np.sum(scission_matrix_sum_y) * n_monomers_groups))
    pos = 0

    sci_pos_arr = np.array(np.where(scission_matrix_sum_y > 0)).transpose()
    progress_bar = tqdm(total=len(sci_pos_arr), position=0)

    for sci_coords in sci_pos_arr:

        x_ind, z_ind = sci_coords
        n_scissions = int(scission_matrix_sum_y[x_ind, z_ind])

        xz0 = mapping.x_centers_2nm[x_ind] * 1e-7, mapping.z_centers_2nm[z_ind] * 1e-7

        for _ in range(n_scissions):
            for _ in range(n_monomers_groups):
                sigma = 20e-7
                # sigma = 1e-4
                x0_gauss = np.random.normal(xz0[0], sigma)
                z0_gauss = np.random.normal(xz0[1], sigma)
                # x_escape_array[pos] = track_monomer(xz0, xx, zz_vac, d_PMMA)
                x_escape_array[pos] = track_monomer([x0_gauss, z0_gauss], xx, zz_vac, d_PMMA)
                pos += 1

        progress_bar.update()

    mon_h_cm = const.V_mon * 1e+7 ** 3 / mapping.step_2nm ** 2 * 1e-7

    x_escape_array_corr = np.zeros(np.shape(x_escape_array))

    for i, x_esc in enumerate(x_escape_array):
        while x_esc > mapping.x_max:
            x_esc -= mapping.l_x
        while x_esc < mapping.x_min:
            x_esc += mapping.l_x
        x_escape_array_corr[i] = x_esc

    x_escape_hist = np.histogram(x_escape_array_corr, bins=mapping.x_bins_2nm * 1e-7)[0]

    delta_zz_vac = x_escape_hist * mon_h_cm

    if double:
        delta_zz_vac = delta_zz_vac + delta_zz_vac[::-1]

    zz_vac_new = zz_vac + delta_zz_vac

    return zz_vac_new


def get_An_tau_arrays(eta, xx, zz_vac, T_C, N):
    hh_vac = 80e-7 - zz_vac
    # eta = rf.get_PMMA_950K_viscosity(T_C)
    gamma = rf.get_PMMA_surface_tension(T_C)
    # N = 100
    h0 = np.trapz(hh_vac, x=xx) / (np.max(xx) - np.min(xx)) * 1e-2  # cm -> m
    l0 = 3.3e-6  # m

    xx_nm = xx * 1e+7
    xx_nm[0] = mapping.x_min
    xx_nm[-1] = mapping.x_max

    An_array = rf.get_An_array(
        xx=xx_nm,
        zz=hh_vac*1e+7,
        l0=l0*1e+9,
        N=N
    ) * 1e-9  # cm -> nm -> m

    tau_n_array = rf.get_tau_n_array(
        eta=eta,
        gamma=gamma,
        h0=h0,
        l0=l0,
        N=N
    )

    return An_array, tau_n_array


def get_An_Bn_tau_arrays(eta, xx, zz_vac, T_C, N):
    hh_vac = 80e-7 - zz_vac
    # eta = rf.get_PMMA_950K_viscosity(T_C)
    gamma = rf.get_PMMA_surface_tension(T_C)
    # N = 100
    h0 = np.trapz(hh_vac, x=xx) / (np.max(xx) - np.min(xx)) * 1e-2  # cm -> m
    l0 = 3.3e-6  # m

    xx_nm = xx * 1e+7
    xx_nm[0] = mapping.x_min
    xx_nm[-1] = mapping.x_max

    An_array = rf.get_An_array(
        xx=xx_nm,
        zz=hh_vac*1e+7,
        l0=l0*1e+9,
        N=N
    ) * 1e-9  # cm -> nm -> m

    Bn_array = rf.get_Bn_array(
        xx=xx_nm,
        zz=hh_vac * 1e+7,
        l0=l0 * 1e+9,
        N=N
    ) * 1e-9  # cm -> nm -> m

    tau_n_array = rf.get_tau_n_array(
        eta=eta,
        gamma=gamma,
        h0=h0,
        l0=l0,
        N=N
    )

    return An_array, Bn_array, tau_n_array


def get_h_at_t_even(xx, An_array, tau_n_array, l0, t):
    return rf.get_h_at_t_even(xx * 1e-2, An_array, tau_n_array, l0, t)


def get_h_at_t(xx, An_array, Bn_array, tau_n_array, l0, t):
    return rf.get_h_at_t(xx * 1e-2, An_array, Bn_array, tau_n_array, l0, t)


# %%
# xx = mapping.x_centers_2nm * 1e-7
# zz_vac = np.zeros(len(xx))
# zip_length = 1000
# d_PMMA = 80e-7
#
# sci_mat = np.load('/Users/fedor/PycharmProjects/MC_simulation/scission_matrix_5e.npy')
#
# zz_vac = get_profile_after_diffusion(sci_mat, zip_length, xx, zz_vac, d_PMMA)

# %%
# plt.figure(dpi=300)
# plt.plot(xx, zz_vac)
# plt.show()

# %%
# t = 0
# T_C = 125
# hh_vac = make_reflow(xx, zz_vac, T_C, 0)