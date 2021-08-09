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

# %%
source = 'data/e_DATA_Pv_80nm/'
n_files_total = 500

D = 20e-6  # C / cm^2
Q = D * mm.area_cm2

# T = 98 C
# weight = 0.25

# T = 118 C
# weight = 0.27

# T = 125 C
weight = 0.275

# T = 150 C
# weight = 0.301

# T = 170 C
# weight = 0.315

# plt.figure(dpi=300)
# plt.plot([125, 150, 170], [0.275, 0.301, 0.315], 'o--')
# plt.grid()
# plt.xlim(90, 180)
# plt.ylim(0.25, 0.32)
# plt.show()

# %%
n_electrons_required = Q / const.e_SI
n_primaries_in_file = 100
n_files_required = int(n_electrons_required / n_primaries_in_file)

# %%
now_z_vac = 0
zz_vac_list = []
zip_lens_list = []
n_monomers_list = []
n_scissions_list = []

scission_matrix_total = np.zeros(mm.hist_10nm_shape)
n_monomers_detached_total = 0

progress_bar = tqdm(total=n_files_required, position=0)

for file_cnt in range(n_files_required):

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

    n_scissions_list. append(np.sum(scission_matrix))
    scission_matrix_total += scission_matrix

    progress_bar.update()


print('Total n_scissions =', np.sum(scission_matrix_total))
