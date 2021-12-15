import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import indexes as ind
from functions import e_matrix_functions as emf
from functions import array_functions as af
from functions import development_functions as df
from functions import MC_functions as mcf
import copy
from collections import deque

af = importlib.reload(af)
df = importlib.reload(df)
emf = importlib.reload(emf)
mcf = importlib.reload(mcf)
ind = importlib.reload(ind)

# %% 1um x 1um
Lx = 1e+3
Ly = 1e+3
D0 = 1e+3

x_min, x_max = -Lx / 2, Lx / 2
z_min, z_max = 0, D0

bin_size = 10

x_bins = np.arange(x_min, x_max + 1, bin_size)
z_bins = np.arange(z_min, z_max + 1, bin_size)

x_centers = (x_bins[:-1] + x_bins[1:]) / 2
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

Q_l = 2e-9  # C / cm
Mn = 2e+5
G = 1.9
Rb = 100
beam_sigma = 100 / 2

n_electrons_required = Q_l * Ly * 1e-7 / 1.6e-19
n_files_required = int(n_electrons_required / 100)

E_dep_matrix = np.zeros((len(x_centers), len(z_centers)))

# %%
progress_bar = tqdm(total=n_files_required, position=0)

n_files = 170
file_cnt = 0

while file_cnt < n_files_required:

    primary_electrons_in_file = 100

    now_e_DATA = np.load('data/4_Atoda/e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
    now_e_DATA = now_e_DATA[np.where(now_e_DATA[:, 7] > 0)]

    if file_cnt > n_files:
        emf.rotate_DATA(now_e_DATA, x_ind=4, y_ind=5)

    emf.add_gaussian_xy_shift_to_e_DATA(
        e_DATA=now_e_DATA,
        x_position=0,
        x_sigma=beam_sigma,
        y_range=[0, 0])

    # af.snake_coord(
    #     array=now_e_DATA,
    #     coord_ind=4,
    #     coord_min=x_min,
    #     coord_max=x_max
    # )

    now_e_DATA_xx = now_e_DATA[:, 4]
    now_e_DATA_zz = now_e_DATA[:, 6]
    now_e_DATA_dE = now_e_DATA[:, 7]

    E_dep_matrix += np.histogramdd(
        sample=[now_e_DATA_xx, now_e_DATA_zz],
        bins=[x_bins, z_bins],
        weights=now_e_DATA_dE
    )[0]

    file_cnt += 1
    progress_bar.update()

# %%
# np.save('notebooks/development/E_dep_atoda_new.npy', E_dep_matrix)
# E_dep_matrix = np.load('notebooks/development/E_dep_atoda_new.npy')

# %%
plt.figure(dpi=300)
# plt.imshow(E_dep_matrix.transpose())
plt.imshow(np.log(E_dep_matrix).transpose())
plt.show()

# %%
rho = 1.19  # g / cc
Na = 6.02e+23

# eps_matrix = E_dep_matrix / 1e-7**3
eps_matrix = E_dep_matrix / bin_size ** 2 / Ly / 1e-21

Mf_matrix = Mn / (1 + G / 100 * eps_matrix * Mn / (rho * Na))

plt.figure(dpi=300)
plt.imshow(np.log(Mf_matrix).transpose())
plt.show()

# %%
# MIBK
# R0 = 51  # A / min
# beta = 3.59e+8  # A / min
# alpha = 1.42

# MIBK : IPA = 1 : 3
R0 = 0.0  # A / min
beta = 1.046e+16
alpha = 3.86

development_rates_A_min = df.get_development_rates_Mf(
    Mf_matrix=Mf_matrix,
    R0=R0,
    alpha=alpha,
    beta=beta
)

development_rates = development_rates_A_min * 0.1 / 60

bin_size = 10  # nm
bin_heights = np.ones(np.shape(development_rates)) * bin_size

# %% Induction times
# ind_times = np.loadtxt('notebooks/development/curves/induction_time.txt')
# # MM, tt = ind_times[:, 0], ind_times[:, 1] * 60
# MM, tt = ind_times[:, 0], ind_times[:, 1]
# # tt[0] = tt[2]
#
# atoda_R = np.loadtxt('notebooks/development/curves/atoda_R.txt')
# atoda_MM = atoda_R[:, 0]
# atoda_RR = atoda_R[:, 1]

# add_time_arr = mcf.log_log_interp(MM, tt)(Mf_matrix[:, 0])

# bin_heights[:, 0] += add_time_arr * development_rates_nm_s[:, 0]

# RR = df.get_development_rates_Mf_arr(
#     Mf_arr=MM,
#     R0=R0,
#     alpha=alpha,
#     beta=beta
# )
#
# plt.figure(dpi=300)
# plt.loglog(MM, RR, 'o')
# plt.loglog(atoda_MM, atoda_RR)
# plt.show()

# %%
now_t = 0

n_cells_removed = 15000

times = np.zeros(n_cells_removed)
profiles = np.zeros((n_cells_removed, len(bin_heights), len(bin_heights[0])))

progress_bar = tqdm(total=n_cells_removed, position=0)

for i in range(n_cells_removed):

    now_n_surface_facets = df.get_n_surface_facets(bin_heights)
    now_surface_inds = np.where(now_n_surface_facets > 0)

    now_development_times = np.zeros(np.shape(now_n_surface_facets))
    now_development_times[now_surface_inds] =\
        bin_heights[now_surface_inds] /\
        development_rates[now_surface_inds] / np.sqrt(now_n_surface_facets[now_surface_inds])

    now_dt = np.min(now_development_times[np.where(now_development_times > 0)])

    bin_heights[now_surface_inds] -=\
        now_dt * development_rates[now_surface_inds] * np.sqrt(now_n_surface_facets[now_surface_inds])

    bin_heights[np.where(np.abs(bin_heights) < 1e-5)] = 0

    now_t += now_dt

    times[i] = now_t
    profiles[i, :, :] = copy.deepcopy(bin_heights)

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.imshow(profiles[1].transpose())
plt.show()

# %%
last_profile = profiles[10, :, :]

inds_x, inds_z = np.where(np.logical_and(last_profile < bin_size, last_profile > 0))

plt.figure(dpi=300)
plt.plot(x_centers[inds_x] / 1000, 1000 - z_centers[inds_z], label='simulation')

plt.xlim(-0.5, 0.5)
plt.ylim(0, 1000)

plt.xlabel('x, um')
plt.ylabel('z, nm')

plt.grid()
# plt.savefig('atoda_96s_MIBK_IPA.jpg', dpi=300)
plt.show()

# %%
np.save('notebooks/development/times_with_induction.npy', times)
np.save('notebooks/development/profiles_with_induction.npy', profiles)

# %%
TT = np.load('notebooks/development/times_WO_induction.npy')
PP = np.load('notebooks/development/profiles_WO_induction.npy')

# TT_i = np.load('notebooks/development/times_with_induction.npy')
# PP_i = np.load('notebooks/development/profiles_with_induction.npy')

TT_i = times
PP_i = profiles

# %%
PP[np.where(PP > 0)] = 1
PP_i[np.where(PP_i > 0)] = 1

# %%
# TT: 1.6 min - 1192; 2.6 min - 1836; 3.1 min - 2386; 3.9 min - 3041; 5.5 min - 4258; 7.8 min - 5379
# TT: 1.6 min - 620; 2.6 min - 1836; 3.1 min - 2386; 3.9 min - 3041; 5.5 min - 4258; 7.8 min - 5379

plt.figure(dpi=300)
plt.imshow(PP[1192].transpose())
# plt.imshow(PP_i[620].transpose())
plt.show()



