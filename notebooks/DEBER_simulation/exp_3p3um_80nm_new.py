import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import constants as const
from mapping import mapping_3p3um_80nm as mm
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import MC_functions as mcf
# from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf
from functions._outdated import SE_functions as ef
from functions import scission_functions as sf
from scipy.signal import medfilt
import indexes as ind

# deber = importlib.reload(deber)
const = importlib.reload(const)
emf = importlib.reload(emf)
mcf = importlib.reload(mcf)
ind = importlib.reload(ind)
mm = importlib.reload(mm)
af = importlib.reload(af)
mf = importlib.reload(mf)
df = importlib.reload(df)
ef = importlib.reload(ef)
rf = importlib.reload(rf)
pf = importlib.reload(pf)
sf = importlib.reload(sf)

# %%
d_PMMA_cm = mm.d_PMMA * 1e-7  # cm
beam_size = 200  # nm
beam_size_cm = beam_size * 1e-7  # cm

t_exp = 320  # s
I_exp = 0.1e-9
frame_lx_cm = 2.6e-1
frame_ly_cm = 2e-1
pitch_cm = 3.3e-4

T_C = 125
Tg = 120
dT = T_C - Tg

y_0 = 3989

n_lines = frame_ly_cm / pitch_cm
line_time = t_exp / n_lines

line_Q = I_exp * line_time
line_n_electrons = line_Q / 1.6e-19
beam_spot_n_electrons = line_n_electrons * mm.ly_cm / frame_lx_cm
beam_spot_n_electrons_1s = beam_spot_n_electrons / t_exp
beam_spot_n_electrons_10s = beam_spot_n_electrons_1s * 10

tau_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/kinetic_curves/arrays/tau.npy')
Mw_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/kinetic_curves/arrays/Mw_125.npy')

# %%
xx = mm.x_bins_5nm  # nm
shape = mm.hist_5nm_shape
bins = mm.bins_5nm
bin_size = mm.step_5nm

zz_vac = np.ones(len(xx)) * 0  # nm
zz_vac_list = [zz_vac]

file_cnt = 0
n_files = 260
primary_electrons_in_file = 100
n_files_10s = 8

zip_length = 456
weight = 0.91  # 125 C
source = '/Volumes/Transcend/new_e_DATA_80nm/'

scission_matrix = np.zeros(shape)
monomer_matrix_2d = np.zeros(np.shape(np.sum(scission_matrix, axis=1)))

time_step = 10  # s

# for i in range(32):
for i in range(1):

    print('##########', i, '##########')

    progress_bar = tqdm(total=n_files_10s, position=0)

    for _ in range(n_files_10s):
        now_DATA = np.load(source + 'e_DATA_' + str(file_cnt % n_files) + '.npy')
        now_DATA = now_DATA[np.where(now_DATA[:, ind.e_DATA_layer_id_ind] <= mm.d_PMMA)]
        now_DATA = now_DATA[np.where(now_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]
        file_cnt += 1

        if file_cnt > n_files:
            emf.rotate_DATA(now_DATA, x_ind=ind.e_DATA_x_ind, z_ind=ind.e_DATA_z_ind)

        for primary_e_id in range(primary_electrons_in_file):

            now_prim_e_DATA = emf.get_e_id_e_DATA(now_DATA, primary_electrons_in_file, primary_e_id)

            # emf.add_uniform_xy_shift_to_e_DATA(now_prim_e_DATA, [-beam_size / 2, beam_size / 2],
            #                                    [-beam_size / 2, beam_size / 2])
            emf.add_gaussian_xy_shift_to_track(now_prim_e_DATA, 0, beam_size / 2, [mm.y_min, mm.y_max])

            af.snake_array(
                array=now_prim_e_DATA,
                x_ind=ind.e_DATA_x_ind,
                y_ind=ind.e_DATA_y_ind,
                z_ind=ind.e_DATA_z_ind,
                xyz_min=[mm.x_min, mm.y_min, -np.inf],
                xyz_max=[mm.x_max, mm.y_max, np.inf]
            )

            for pos, line in enumerate(now_prim_e_DATA):  # delete events that are out of resist layer

                now_x, now_z = line[ind.e_DATA_x_ind], line[ind.e_DATA_z_ind]

                if now_z < mcf.lin_lin_interp(xx, zz_vac)(now_x):
                    now_prim_e_DATA[pos, ind.e_DATA_process_id_ind] = 0  # change type to simulate zz_vac

            now_prim_e_val_DATA = \
                now_prim_e_DATA[np.where(now_prim_e_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

            scission_probs = np.ones(len(now_prim_e_val_DATA)) * weight
            scission_arr = np.array(np.random.random(len(scission_probs)) < scission_probs).astype(int)

            scission_matrix += np.histogramdd(
                sample=now_prim_e_val_DATA[:, ind.e_DATA_coord_inds],
                bins=bins,
                weights=scission_arr
            )[0]

        progress_bar.update()

# %% per 1 nm !!!
scission_matrix = np.load('notebooks/DEBER_simulation/scission_matrix_10s_5nm.npy')

print('scission matrix is obtained, sum =', np.sum(scission_matrix))

scission_array = np.sum(np.average(scission_matrix, axis=1), axis=1) / mm.step_5nm

print('scission array sum =', np.sum(scission_array))

plt.figure(dpi=300)
plt.plot(mm.x_centers_5nm, scission_array)
plt.show()

# %% move scission array to 50nm bins
xx_5nm_coords = mm.x_centers_5nm
weights = scission_array

scission_array_50nm = np.histogram(xx_5nm_coords, bins=mm.x_bins_50nm, weights=scission_array)[0]

plt.figure(dpi=300)
# plt.plot(mm.x_centers_5nm, weights)
plt.plot(mm.x_centers_50nm, scission_array_50nm / np.max(scission_array_50nm))
plt.plot(mm.x_centers_5nm, scission_array / np.max(scission_array))
plt.show()

# %% per 1 nm !!!
xx_50nm = mm.x_bins_50nm
zz_vac_50nm = mcf.lin_lin_interp(xx, zz_vac)(mm.x_bins_50nm)
vac_area_50nm = np.zeros(len(scission_array_50nm))

for j in range(len(vac_area_50nm)):
    vac_area_50nm[j] = (zz_vac_50nm[j] + zz_vac_50nm[j + 1]) / 2 * mm.step_50nm

resist_area_50nm = np.ones(len(vac_area_50nm)) * mm.d_PMMA * mm.step_50nm - vac_area_50nm
resist_volume_50nm = resist_area_50nm * 1  # nm^3 per nm
resist_n_monomers_50nm = resist_volume_50nm / const.V_mon_nm3


vac_area = np.zeros(len(scission_array))

for j in range(len(vac_area)):
    vac_area[j] = (zz_vac[j] + zz_vac[j + 1]) / 2 * mm.step_5nm

resist_area = np.ones(len(vac_area)) * mm.d_PMMA * mm.step_5nm - vac_area
resist_volume = resist_area * 1  # nm^3 per nm
resist_n_monomers = resist_volume / const.V_mon_nm3


# %%
i = 0
k_s_exp_50nm = scission_array_50nm / resist_n_monomers_50nm / ((i + 1) * time_step)

tau_exp_array_50nm = y_0 * k_s_exp_50nm * time_step * (i + 1)
Mw_array_50nm = np.zeros(len(tau_exp_array_50nm))

for j in range(len(Mw_array_50nm)):
    tau_ind = np.argmin(np.abs(tau_125 - tau_exp_array_50nm[j]))
    Mw_array_50nm[j] = Mw_125[tau_ind]

Mw_array_filt_50nm = medfilt(Mw_array_50nm, kernel_size=3)

# mobs_array = rf.move_Mw_to_mob(T_C, Mw_array_50nm)

k_s_exp = scission_array / resist_n_monomers / ((i + 1) * time_step)

tau_exp_array = y_0 * k_s_exp * time_step * (i + 1)
Mw_array = np.zeros(len(tau_exp_array))

for j in range(len(Mw_array)):
    tau_ind = np.argmin(np.abs(tau_125 - tau_exp_array[j]))
    Mw_array[j] = Mw_125[tau_ind]


plt.figure(dpi=300)
plt.plot(mm.x_centers_5nm, Mw_array)
plt.plot(mm.x_centers_50nm, Mw_array_50nm)
plt.plot(mm.x_centers_50nm, Mw_array_filt_50nm)
plt.show()


# %% back to 5 nm
xx_50_compl = np.concatenate(([mm.x_min], mm.x_centers_50nm, [mm.x_max]))
Mw_array_filt_50nm_compl = np.concatenate(([Mw_array_filt_50nm[0]], Mw_array_filt_50nm, [Mw_array_filt_50nm[-1]]))
Mw_array_5nm = mcf.lin_lin_interp(xx_50_compl, Mw_array_filt_50nm_compl)(mm.x_centers_5nm)

Mw_array_filt_5nm = medfilt(Mw_array_5nm, kernel_size=23)
mobs_array_5nm = rf.move_Mw_to_mob(T_C, Mw_array_filt_5nm)

Mw_array_filt = medfilt(Mw_array, kernel_size=13)
mobs_array = rf.move_Mw_to_mob(T_C, Mw_array_filt)

plt.figure(dpi=300)
plt.plot(mm.x_centers_5nm, mobs_array_5nm, '.')
plt.plot(mm.x_centers_5nm, mobs_array)
plt.show()

# %%
now_monomer_matrix_2d = np.average(scission_matrix, axis=1) * zip_length
monomer_matrix_2d += now_monomer_matrix_2d

print('simulate diffusion ...')

monomer_portion = 10

monomer_matrix_2d_final = df.track_all_monomers(
    monomer_matrix_2d=monomer_matrix_2d,
    xx=xx,
    zz_vac=zz_vac,
    d_PMMA=mm.d_PMMA,
    dT=dT,
    wp=1,
    t_step=time_step,
    dt=0.5,
    n_portion=monomer_portion
)

print('diffusion is simulated')

# %%
monomer_matrix_2d_final = np.load('notebooks/DEBER_simulation/monomer_matrix_2d_final.npy')

# %%
monomer_array_final = monomer_matrix_2d_final[:, 0]
monomer_array_final_filt = medfilt(monomer_array_final, kernel_size=13)
delta_z_array = monomer_array_final_filt * const.V_mon_nm3 / bin_size / mm.ly

plt.figure(dpi=300)
# plt.plot(mm.x_centers_5nm, monomer_array_final)
# plt.plot(mm.x_centers_5nm, monomer_array_final_filt)
plt.plot(mm.x_centers_5nm, delta_z_array)
plt.title('evaporated monomer')
plt.show()

# %% get profile before evolver
zz_evolver = mm.d_PMMA - delta_z_array

xx_final_tochno = np.concatenate(([mm.x_min], mm.x_centers_5nm, [mm.x_max]))
zz_final_tochno = np.concatenate(([zz_evolver[0]], zz_evolver, [zz_evolver[-1]]))
mobs_array_final_tochno = np.concatenate(([mobs_array[0]], mobs_array_5nm, [mobs_array[-1]]))

plt.figure(dpi=300)
plt.plot(xx_final_tochno, zz_final_tochno)

plt.title('profile before SE, i = ' + str(i))
plt.show()

# np.save('notebooks/DEBER_simulation/xx_evolver_nm.npy', xx_final_tochno)
# np.save('notebooks/DEBER_simulation/zz_evolver_nm.npy', zz_final_tochno)
# np.save('notebooks/DEBER_simulation/mobs_evolver.npy', mobs_array_final_tochno)

# %%
ef.create_datafile_non_period(xx_final_tochno, zz_final_tochno, mobs_array_final_tochno)
