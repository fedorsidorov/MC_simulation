import importlib
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import matplotlib.pyplot as plt
import MC_classes as mcc
import constants as const
import indexes as ind
from mapping import mapping_3p3um_80nm as mapping
from functions import array_functions as af
from functions import diffusion_functions as df
from functions import e_matrix_functions as emf
from functions import reflow_functions as rf
from functions import scission_functions as sf
from functions import plot_functions as pf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcc = importlib.reload(mcc)
af = importlib.reload(af)
df = importlib.reload(df)
rf = importlib.reload(rf)
sf = importlib.reload(sf)
pf = importlib.reload(pf)


# %%
def get_e_DATA_PMMA_val(xx, zz_vac, d_PMMA, n_electrons, E0, r_beam):
    ly = mapping.l_y * 1e-7

    structure = mcc.Structure(
        d_PMMA=d_PMMA,
        xx=xx,
        zz_vac=zz_vac,
        ly=ly)

    simulator = mcc.Simulator(
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

    return e_DATA, e_DATA_PMMA_val


def get_e_DATA_Pn(xx, zz_vac, d_PMMA, n_electrons, E0, r_beam):
    ly = mapping.l_y * 1e-7

    structure = mcc.Structure(
        d_PMMA=d_PMMA,
        xx=xx,
        zz_vac=zz_vac,
        ly=ly)

    simulator = mcc.Simulator(
        structure=structure,
        n_electrons=n_electrons,
        E0_eV=E0,
        r_beam=r_beam
    )
    simulator.prepare_e_deque()
    simulator.start_simulation()

    e_DATA = simulator.get_total_history()

    e_DATA_Pn = e_DATA[np.where(np.logical_and(
        e_DATA[:, ind.DATA_layer_id_ind] == ind.PMMA_ind,
        e_DATA[:, ind.DATA_process_id_ind] > 0)
    )]

    return e_DATA, e_DATA_Pn


def get_scission_matrix_degpaths(e_DATA_PMMA_val, weight):
    deg_paths = sf.degpaths_all_WO_Oval

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


def get_scission_matrix(e_DATA, weight):

    af.snake_array(
        array=e_DATA,
        x_ind=ind.DATA_x_ind,
        y_ind=ind.DATA_y_ind,
        z_ind=ind.DATA_z_ind,
        xyz_min=[mapping.x_min, mapping.y_min, -np.inf],
        xyz_max=[mapping.x_max, mapping.y_max, np.inf]
    )

    # deg_paths = sf.degpaths_all_WO_Oval
    deg_paths = sf.degpaths_all

    e_DATA_PMMA = e_DATA[np.where(e_DATA[:, ind.DATA_layer_id_ind] == ind.PMMA_ind)]
    e_DATA_PMMA_val = e_DATA_PMMA[np.where(e_DATA_PMMA[:, ind.DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

    e_matrix_E_dep = np.histogramdd(
        sample=e_DATA_PMMA[:, ind.DATA_coord_inds],
        bins=mapping.bins_5nm,
        weights=e_DATA_PMMA[:, ind.DATA_E_dep_ind]
    )[0]

    scissions = sf.get_scissions_easy(e_DATA_PMMA_val, weight=weight)

    e_matrix_val_sci = np.histogramdd(
        sample=e_DATA_PMMA_val[:, ind.DATA_coord_inds],
        bins=mapping.bins_5nm,
        weights=scissions
    )[0]

    return e_matrix_val_sci, e_matrix_E_dep


# %%
# xx = mapping.x_centers_5nm * 1e-7
# zz_vac = np.zeros(len(xx)) + 5e-7
# d_PMMA = 80e-7  # cm
#
# data, data_val = get_e_DATA_PMMA_val(xx, zz_vac, d_PMMA, 10, E0=20e+3, r_beam=150e-7)
#
# # %%
# pf.plot_e_DATA(data, d_PMMA=80, E_cut=5, proj='xz')
