import numpy as np
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mapping import mapping_viscosity_80nm as mm
from functions import mapping_functions as mf
from functions import e_matrix_functions as emf
from functions import array_functions as af
from functions import scission_functions as sf
import constants as const
import indexes as ind

mapping = importlib.reload(mm)
const = importlib.reload(const)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
sf = importlib.reload(sf)
mf = importlib.reload(mf)

# %% j = 80e-9  # A / cm^2
xx = mm.x_centers_10nm
zz_vac = np.zeros(len(xx))

source = 'data/e_DATA_Pv_80nm/'
n_files_total = 500

D = 20e-6  # C / cm^2
total_time = 250  # s
Q = D * mm.area_cm2

# T = 125 C
weight = 0.275

n_electrons_required = Q / const.e_SI
n_files_required = int(n_electrons_required / 100)

n_primaries_in_file = 100

# %%
resist_matrix = np.load('/Volumes/ELEMENTS/chains_viscosity_80nm/10nm/resist_matrix_1.npy')
chain_lens = np.load('/Volumes/ELEMENTS/chains_viscosity_80nm/10nm/prepared_chains_1/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Volumes/ELEMENTS/chains_viscosity_80nm/10nm/chain_tables_1/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_10nm_shape

# %%
now_z_vac = 0
zz_vac_list = []
zip_lens_list = []
n_monomers_list = []
n_scissions_list = []

scission_matrix_total = np.zeros(mm.hist_10nm_shape)
n_monomers_detached_total = 0

# progress_bar = tqdm(total=n_files_required, position=0)

for file_cnt in range(1):
# for file_cnt in range(n_files_required):

    print(file_cnt)

    now_e_DATA_Pv = np.load(source + 'e_DATA_' + str(file_cnt % n_files_total) + '.npy')

    now_e_DATA_Pv = now_e_DATA_Pv[np.where(now_e_DATA_Pv[:, ind.e_DATA_z_ind] > now_z_vac)]

    emf.add_individual_uniform_xy_shifts_to_e_DATA(now_e_DATA_Pv, n_primaries_in_file,
                                                   [-mm.lx/2, mm.lx/2], [-mm.ly/2, mm.ly/2])

    af.snake_array(
        array=now_e_DATA_Pv,
        x_ind=ind.e_DATA_x_ind,
        y_ind=ind.e_DATA_y_ind,
        z_ind=ind.e_DATA_z_ind,
        xyz_min=[mm.x_min, mm.y_min, -np.inf],
        xyz_max=[mm.x_max, mm.y_max, np.inf]
    )

    scission_matrix = np.histogramdd(
        sample=now_e_DATA_Pv[:, ind.e_DATA_coord_inds],
        bins=mm.bins_10nm,
        weights=sf.get_scissions(now_e_DATA_Pv, weight=weight)
    )[0]

    n_scissions_list.append(np.sum(scission_matrix))
    scission_matrix_total += scission_matrix

    # print('mapping ...')
    mf.process_mapping_NEW(scission_matrix, resist_matrix, chain_tables)

    # print('depolymerization ...')
    n_monomers_detached = mf.process_depolymerization_NEW(resist_matrix, chain_tables)
    n_monomers_list.append(n_monomers_detached)
    n_monomers_detached_total += n_monomers_detached

    V_free = n_monomers_detached * const.V_mon_cm3
    delta_z = V_free / (mm.lx * mm.ly * 1e-14) * 1e+7
    now_z_vac += delta_z
    zz_vac_list.append(now_z_vac)

# %%
n_chains_completed = 0

for i, ct in enumerate(chain_tables):
    if len(np.where(ct[:, -1] == 10)[0]) == 0:
        print(i)
        n_chains_completed += 1

# %%
data = np.loadtxt('notebooks/kinetic_curves/data/kin_curve_170C_80nm.txt')

plt.figure(dpi=300)
# plt.plot(zip_lens_list)

dose_list_depol = np.arange(n_files_required) * n_primaries_in_file * const.e_SI / mm.area_cm2 * 1e+6
L_norm_depol = (80 - np.array(zz_vac_list)) / 80

plt.plot(dose_list_depol, L_norm_depol)
plt.plot(data[:, 0], data[:, 1], 'o-')

plt.grid()

plt.show()

# %%
scission_matrix_2d = np.sum(scission_matrix_total, axis=1)


