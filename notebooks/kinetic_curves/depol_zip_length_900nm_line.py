import importlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import plot_functions as pf
from functions import G_functions as Gf
import constants as const
from mapping import mapping_viscosity_900nm as mm

const = importlib.reload(const)
mm = importlib.reload(mm)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)
Gf = importlib.reload(Gf)

# %% estimate scission number
xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))
# zz_vac = np.ones(len(xx)) * ((1 + np.cos(xx * np.pi / (lx / 2))) * d_PMMA) / 5

source = 'data/e_DATA_Pv_900nm/'

# %% dose 1 - 0.05 uC/cm^2
scission_matrix_total = np.zeros(mm.hist_5nm_shape)

# T = 160 C
weight = 0.256

# dose 1 - 59998 scissions, A = 381526 nm^2
# n_electrons_per_nm = 15.625
# delta_h = 200  # nm
# total_time = 59

# dose 2 - 180169 scissions, A = 1275148 nm^2
# n_electrons_per_nm = 62.5
# delta_h = 700  # nm
# total_time = 235  # s

# dose 3 - 648048 scissions, A = 2037395 nm^2
n_electrons_per_nm = 271.875
delta_h = 900  # nm
total_time = 1024  # s

n_electrons_required = int(n_electrons_per_nm * mm.ly)

n_primaries_in_file = 100

n_files_required = int(n_electrons_required / n_primaries_in_file)
t_step = total_time / n_files_required

progress_bar = tqdm(total=n_files_required, position=0)


def get_zz_vac(t):
    return delta_h * t / total_time


for file_cnt in range(n_files_required):

    now_e_DATA_Pv = np.load(source + 'e_DATA_' + str(file_cnt % n_files_required) + '.npy')

    now_time = file_cnt * t_step
    now_z_vac = get_zz_vac(now_time)

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
        bins=mm.bins_5nm,
        weights=sf.get_scissions(now_e_DATA_Pv, weight=weight)
    )[0]

    scission_matrix_total += scission_matrix

    progress_bar.update()

print(np.sum(scission_matrix_total))

# %%
# event_matrix_total_2d = np.sum(event_matrix_total, axis=1)
# event_matrix_total_2d = np.sum(event_matrix_total[:, :, -10:], axis=2)
# event_matrix_total_2d = np.sum(event_matrix_total[:, :, -10:], axis=2)

scission_matrix_total_1d = np.sum(np.sum(scission_matrix_total, axis=1), axis=0)

# %%
plt.figure(dpi=300)
# plt.imshow(event_matrix_total_2d.transpose())
plt.plot(mm.z_centers_5nm, scission_matrix_total_1d, 'o-')
plt.show()

# %%
# dose 1
# n_scissions = 59998
# A = 381526

# dose 2
# n_scissions = 180169
# A = 1275148

# dose 3
n_scissions = 648048
A = 2037395

V_exp = A * mm.ly
n_mon_escaped_exp = V_exp / const.V_mon_nm3

zip_length_exp = n_mon_escaped_exp / n_scissions

print(zip_length_exp)
