import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_3p3um_80nm as mm
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf

import indexes as ind

const = importlib.reload(const)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mm = importlib.reload(mm)
af = importlib.reload(af)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
lx = mm.lx * 1e-7
ly = mm.ly * 1e-7
area = lx * ly

d_PMMA = mm.z_max
d_PMMA_cm = d_PMMA * 1e-7  # cm

j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * lx
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * lx
t = dose_l / j_exp_l  # 316 s
dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))

T_C = 125
Tg = 120
dT = T_C - Tg

time_s = 10

y_0 = 3989

tau_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mw_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Boyd_kinetic_curves/arrays/Mw_125.npy') * 100

# %%
xx = mm.x_bins_10nm  # nm
zz_vac = np.ones(len(xx)) * 0  # nm
zz_vac_list = [zz_vac]

file_cnt = 0
n_files = 3200
primary_electrons_in_file = 10

# zip_length = 1000
zip_length = 456
r_beam = 100
weight = 0.275  # 125 C
# weight = 0.3
# source = 'data/e_DATA_Pv_80nm/'
source = '/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA_Pn_80nm_point/'

scission_matrix = np.zeros(mm.hist_10nm_shape)
monomer_matrix_2d = np.zeros(np.shape(np.sum(scission_matrix, axis=1)))

time_step = 10  # s

Mw_matrix = np.zeros((32, 330))

for i in range(32):

    # print('!!!!!!!!!', i, '!!!!!!!!!')

    for _ in range(8):
        now_DATA = np.load(source + 'e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
        file_cnt += 1

        if file_cnt > n_files:
            emf.rotate_DATA(now_DATA)

        for primary_e_id in range(primary_electrons_in_file):

            now_prim_e_DATA = emf.get_e_id_e_DATA(now_DATA, primary_electrons_in_file, primary_e_id)

            emf.add_gaussian_xy_shift_to_track(now_prim_e_DATA, 0, r_beam, [mm.y_min, mm.y_max])

            af.snake_array(
                array=now_prim_e_DATA,
                x_ind=ind.e_DATA_x_ind,
                y_ind=ind.e_DATA_y_ind,
                z_ind=ind.e_DATA_z_ind,
                xyz_min=[mm.x_min, mm.y_min, -np.inf],
                xyz_max=[mm.x_max, mm.y_max, np.inf]
            )

            now_prim_e_val_DATA = \
                now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

            scission_matrix += np.histogramdd(
                sample=now_prim_e_val_DATA[:, ind.e_DATA_coord_inds],
                bins=mm.bins_10nm,
                weights=sf.get_scissions(now_prim_e_val_DATA, weight=weight)
            )[0]

    # print('scission matrix is obtained, sum =', np.sum(scission_matrix))

    scission_array = np.sum(np.average(scission_matrix, axis=1), axis=1)

    # print('scission array sum =', np.sum(scission_array))

    # now it is all about 10nm-depth profile !!!

    zz_vac_area = np.zeros(len(scission_array))

    for ii in range(len(zz_vac_area)):
        zz_vac_area[ii] = (zz_vac[ii] + zz_vac[ii + 1]) / 2 * mm.step_10nm

    resist_area = np.ones(len(zz_vac_area)) * d_PMMA * mm.step_10nm - zz_vac_area
    resist_volume = resist_area * mm.step_10nm  # nm^3
    resist_n_monomers = resist_volume / const.V_mon_nm3

    k_s_exp = scission_array / resist_n_monomers / ((i + 1) * time_step)

    tau_exp_array = y_0 * k_s_exp * (i * time_step)
    Mw_array = np.zeros(len(tau_exp_array))

    for ii in range(len(Mw_array)):
        tau_ind = np.argmin(np.abs(tau_125 - tau_exp_array[ii]))
        Mw_array[ii] = Mw_125[tau_ind]

    Mw_matrix[i, :] = Mw_array

    # mobs_array = rf.move_Mw_to_mob(T_C, Mw_array)

    # scission_array_100nm = df.move_10nm_to_100nm(scission_array)
    # sci_fit_params = rf.get_fit_params_sci_arr(mm.x_centers_100nm, scission_array_100nm)
    # scission_array_50nm_fit = rf.gauss(mm.x_centers_50nm, *sci_fit_params)

# %%
tt = np.arange(0, 320, 10)
Mw = Mw_matrix[:, 164]

fig, ax = plt.subplots(dpi=600)
# fig = plt.gcf()
fig.set_size_inches(3.5, 3.5)

font_size = 8

# plt.figure(dpi=300)

plt.plot(tt, Mw)
# plt.semilogy(tt, rf.get_viscosity_W(125, Mw))

# ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

# plt.legend(fontsize=font_size)
plt.xlim(0, 300)
# plt.ylim(40, 80)
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.xlabel('$x$, нм', fontsize=font_size)
plt.ylabel('$z$, нм', fontsize=font_size)
plt.grid()

# plt.show()
plt.savefig('Mw.tiff', dpi=600)




