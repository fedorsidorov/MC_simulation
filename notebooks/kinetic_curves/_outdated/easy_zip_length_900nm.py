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

# %%
xx = mm.x_centers_10nm
zz_vac = np.zeros(len(xx))
# zz_vac = np.ones(len(xx)) * ((1 + np.cos(xx * np.pi / (lx / 2))) * d_PMMA) / 5

source = 'data/e_DATA_Pv_900nm/'
n_files_total = 500

D = 0.9e-6  # C / cm^2
Q = D * mm.area_cm2

n_electrons_required = Q / const.e_SI
# n_files_required = int(n_electrons_required / 100)
n_files_required = 6

# 160 C
data = np.loadtxt('data/kinetic_curves/kin_curve_160C_900nm.txt')
weight = 0.31
# zip_length = 4500  # from initial slope
zip_length = 5500  # from end point

n_primaries_in_file = 100

progress_bar = tqdm(total=n_files_required, position=0)

now_z_vac = 0
scission_matrix_total = np.zeros(mm.hist_10nm_shape)

dose_list = [0]
L_norm_list = [1]

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

    scission_matrix_total += scission_matrix

    n_monomers = np.sum(scission_matrix) * zip_length
    V_monomers = n_monomers * const.V_mon_cm3

    delta_z_cm = V_monomers / mm.area_cm2
    delta_z = delta_z_cm * 1e+7

    now_z_vac += delta_z

    now_Q = (file_cnt + 1) * n_primaries_in_file * 1.6e-19
    now_dose = now_Q / mm.area_cm2

    dose_list.append(now_dose * 1e+6)
    L_norm_list.append(1 - now_z_vac / mm.d_PMMA)

    progress_bar.update()


# %%
plt.figure(dpi=300)

plt.plot(dose_list, L_norm_list, label='sim, zip length = ' + str(zip_length))
plt.plot(data[:, 0], data[:, 1], 'o-', label='experiment')
plt.xlabel('D, $\mu$C/cm$^2$')
plt.ylabel('L/L$_0$')
# plt.xlim(0, 20)
# plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.show()

# %%
np.save('notebooks/kinetic_curves/sim_data/dose_list_900nm_new.npy', dose_list)
np.save('notebooks/kinetic_curves/sim_data/L_norm_160C_5500_new.npy', L_norm_list)
