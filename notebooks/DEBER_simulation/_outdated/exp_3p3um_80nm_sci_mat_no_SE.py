import importlib
import matplotlib.pyplot as plt
import numpy as np
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
total_scission_array = np.sum(total_scission_matrix_2d, axis=1)

# np.save('notebooks/DEBER_simulation/toal_scission_array.npy', total_scission_array)

plt.figure(dpi=300)
plt.plot(total_scission_array)
plt.show()

# %%

print('now n scissions =', np.sum(scission_matrix_2d))

monomer_matrix_2d = scission_matrix_2d * zip_length

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

monomer_array_final = monomer_matrix_2d_final[:, 0]
monomer_array_final_filt = medfilt(monomer_array_final, kernel_size=13)
dz_array = monomer_array_final_filt * const.V_mon_nm3 / bin_size / mm.ly

xx_compl = np.concatenate(([mm.x_bins_5nm[0]], mm.x_centers_5nm, [mm.x_bins_5nm[-1]]))
dz_compl = np.concatenate(([dz_array[0]], dz_array, [dz_array[-1]]))

dz_vac = mcf.lin_lin_interp(xx_compl, dz_compl)(xx)
zz_vac += dz_vac
zz_vac_matrix[n] = zz_vac

# plt.figure(dpi=300)
# plt.plot(mm.x_bins_5nm, 80 - zz_vac)
# plt.title('profile after monomer evaporation')
# plt.show()


# %%
plt.figure(dpi=300)

for i in range(len(zz_vac_matrix)):
    plt.plot(xx, 80 - zz_vac_matrix[i, :])


plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()
plt.show()
