import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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
from functions import SE_functions as ef
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

# %%
xx = mm.x_bins_5nm  # nm
shape = mm.hist_5nm_shape
bins = mm.bins_5nm
bin_size = mm.step_5nm

zz_vac = np.ones(len(xx)) * 0  # nm
zz_vac_matrix = np.zeros((32, len(xx)))

file_cnt = 0
n_files = 260
primary_electrons_in_file = 100
n_files_10s = 8

zip_length = 456

total_scission_matrix_2d = np.zeros((660, 16))
scission_matrix_4d = np.load('notebooks/DEBER_simulation/scission_matrix_4d.npy')

time_step = 10  # s

for n in range(32):

    print('##########', n, '##########')

    scission_matrix_3d = scission_matrix_4d[n, :, :, :]
    scission_matrix_2d = np.average(scission_matrix_3d, axis=1)

    for i in range(len(scission_matrix_2d)):
        for j in range(len(scission_matrix_2d[0])):

            if scission_matrix_2d[i, j] > 0:
                now_x, now_z = mm.x_centers_5nm[i], mm.z_centers_5nm[j]

                if now_z < mcf.lin_lin_interp(xx, zz_vac)(now_x):
                    scission_matrix_2d[i, j] = 0

    total_scission_matrix_2d += scission_matrix_2d

# %%
total_scission_matrix_2d = np.around(total_scission_matrix_2d).astype(int)

total_scission_array = np.load('notebooks/DEBER_simulation/toal_scission_array.npy')
total_scission_array = np.around(total_scission_array).astype(int)

# plt.figure(dpi=300)
# plt.plot(total_scission_array)
# plt.show()

# %%
monomer_matrix_2d = total_scission_matrix_2d * zip_length

# %%
monomer_matrix_2d_final = np.zeros(np.shape(monomer_matrix_2d))
half_angle = np.pi / 12

progress_bar = tqdm(total=len(monomer_matrix_2d), position=0)

for i in range(len(monomer_matrix_2d)):
    for j in range(len(monomer_matrix_2d[0])):

        if monomer_matrix_2d[i, j] == 0:
            continue

        for _ in range(monomer_matrix_2d[i, j]):
            now_x, now_z = mm.x_centers_5nm[i], mm.z_centers_5nm[j]

            delta_x = 2 * now_z * np.sin(half_angle)
            x_min, x_max = now_x - delta_x / 2, now_x + delta_x / 2

            x_inds = np.where(np.logical_and(
                mm.x_centers_5nm >= x_min, mm.x_centers_5nm <= x_max
            ))[0]

            now_x_ind = np.random.choice(x_inds)
            monomer_matrix_2d_final[now_x_ind] += 1

    progress_bar.update()

# %%
monomer_array_final = monomer_matrix_2d_final[:, 0]
# np.save('notebooks/DEBER_simulation/scission_matrix_10s_5nm.npy', monomer_array_final)
monomer_array_final = np.load('notebooks/DEBER_simulation/scission_matrix_10s_5nm.npy')

plt.figure(dpi=300)
plt.plot(monomer_array_final)
plt.show()

# %%
monomer_array_final = monomer_matrix_2d_final[:, 0]
monomer_array_final_filt = medfilt(monomer_array_final, kernel_size=13)
dz_array = monomer_array_final_filt * const.V_mon_nm3 / bin_size / mm.ly

xx_compl = np.concatenate(([mm.x_bins_5nm[0]], mm.x_centers_5nm, [mm.x_bins_5nm[-1]]))
dz_compl = np.concatenate(([dz_array[0]], dz_array, [dz_array[-1]]))

dz_vac = mcf.lin_lin_interp(xx_compl, dz_compl)(xx)
zz_vac += dz_vac
# zz_vac_matrix[n] = zz_vac

plt.figure(dpi=300)
plt.plot(mm.x_bins_5nm, 80 - zz_vac)
plt.title('profile after monomer evaporation')
plt.show()

# %%
xx_5nm_coords = mm.x_centers_5nm
weights = total_scission_array

scission_array_50nm = np.histogram(xx_5nm_coords, bins=mm.x_bins_50nm, weights=weights)[0]

plt.figure(dpi=300)
# plt.plot(mm.x_centers_5nm, weights)
plt.plot(mm.x_centers_50nm, scission_array_50nm / np.max(scission_array_50nm))
# plt.plot(mm.x_centers_5nm, scission_array / np.max(scission_array))
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

vac_area = np.zeros(len(total_scission_array))

for j in range(len(vac_area)):
    vac_area[j] = (zz_vac[j] + zz_vac[j + 1]) / 2 * mm.step_5nm

resist_area = np.ones(len(vac_area)) * mm.d_PMMA * mm.step_5nm - vac_area
resist_volume = resist_area * 1  # nm^3 per nm
resist_n_monomers = resist_volume / const.V_mon_nm3

# %%
tau_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/kinetic_curves/arrays/tau.npy')
Mw_125 = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/kinetic_curves/arrays/Mw_125.npy')

i = 31
k_s_exp_50nm = scission_array_50nm / resist_n_monomers_50nm / ((i + 1) * time_step)

tau_exp_array_50nm = y_0 * k_s_exp_50nm * time_step * (i + 1)
Mw_array_50nm = np.zeros(len(tau_exp_array_50nm))

for j in range(len(Mw_array_50nm)):
    tau_ind = np.argmin(np.abs(tau_125 - tau_exp_array_50nm[j]))
    Mw_array_50nm[j] = Mw_125[tau_ind]

Mw_array_filt_50nm = medfilt(Mw_array_50nm, kernel_size=3)

k_s_exp = total_scission_array / resist_n_monomers / ((i + 1) * time_step)

tau_exp_array = y_0 * k_s_exp * time_step * (i + 1)
Mw_array = np.zeros(len(tau_exp_array))

for j in range(len(Mw_array)):
    tau_ind = np.argmin(np.abs(tau_125 - tau_exp_array[j]))
    Mw_array[j] = Mw_125[tau_ind]


plt.figure(dpi=300)
# plt.plot(mm.x_centers_5nm, Mw_array)
plt.plot(mm.x_centers_50nm, Mw_array_50nm)
# plt.plot(mm.x_centers_50nm, Mw_array_filt_50nm)
plt.show()


# %% back to 5 nm
xx_50_compl = np.concatenate(([mm.x_min], mm.x_centers_50nm, [mm.x_max]))
Mw_array_filt_50nm_compl = np.concatenate(([Mw_array_filt_50nm[0]], Mw_array_filt_50nm, [Mw_array_filt_50nm[-1]]))
Mw_array_5nm = mcf.lin_lin_interp(xx_50_compl, Mw_array_filt_50nm_compl)(mm.x_centers_5nm)

Mw_array_filt_5nm = medfilt(Mw_array_5nm, kernel_size=23)
mobs_array_5nm = rf.move_Mw_to_mob(T_C, Mw_array_filt_5nm)

Mw_array_filt = medfilt(Mw_array, kernel_size=13)
mobs_array = rf.move_Mw_to_mob(T_C, Mw_array_filt)

mobs_sin = (1 + np.cos(mm.x_centers_5nm / mm.lx * 2 * np.pi)) * np.max(mobs_array) / 2

Mw_sin = (1 - np.cos(mm.x_centers_5nm / mm.lx * 2 * np.pi)) * 250 / 2 + 20

plt.figure(dpi=300)
# plt.plot(mm.x_centers_5nm, Mw_array_5nm)
# plt.plot(mm.x_centers_5nm, Mw_sin)
plt.plot(mm.x_centers_5nm, mobs_array_5nm, '.')
plt.plot(mm.x_centers_5nm, mobs_sin)
plt.show()

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
zz_final_tochno = np.concatenate(([zz_evolver[0]], zz_evolver, [zz_evolver[-1]])) * 2
zz_final_tochno = zz_final_tochno - 80

mobs_array_final_tochno = np.concatenate(([mobs_array[0]], mobs_array_5nm, [mobs_array[-1]]))
# mobs_array_final_tochno = np.concatenate(([mobs_sin[0]], mobs_sin, [mobs_sin[-1]]))

xx_final_tochno_50nm = mm.x_bins_50nm
zz_final_tochno_50nm = mcf.lin_lin_interp(xx_final_tochno, zz_final_tochno)(mm.x_bins_50nm)

mobs_array_final_tochno_50nm = mcf.lin_lin_interp(xx_final_tochno, mobs_array_final_tochno)(mm.x_bins_50nm)
# mobs_array_final_tochno_50nm = np.ones(len(xx_final_tochno_50nm)) * 1e+1
# mobs_array_final_tochno_50nm[30:-30] = 1e+4

plt.figure(dpi=300)
# plt.plot(xx_final_tochno, zz_final_tochno)
# plt.plot(xx_final_tochno_50nm, zz_final_tochno_50nm)
# plt.plot(xx_final_tochno, mobs_array_final_tochno)
plt.plot(xx_final_tochno_50nm, mobs_array_final_tochno_50nm)
plt.title('profile before SE, i = ' + str(i))
plt.show()

# %%
# ef.create_datafile_non_period(xx_final_tochno, zz_final_tochno, mobs_array_final_tochno)
# ef.create_datafile_no_mob_fit(xx_final_tochno, zz_final_tochno, mobs_array_final_tochno)
ef.create_datafile_no_mob_fit(xx_final_tochno_50nm, zz_final_tochno_50nm, mobs_array_final_tochno_50nm)

# %%
profile = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SIM/vlist_SIM.txt')

plt.figure(dpi=300)
plt.plot(profile[:, 1] * 1000, profile[:, 2] * 1000, 'o')
plt.ylim(40, 100)
plt.show()

# %% Photoabsorption cross-section
u_ph = 1 / ((6.3e+5 * 5 + 1.22e+6 * 8 + 4.97e+5 * 2) * const.M_mon * const.n_MMA)




