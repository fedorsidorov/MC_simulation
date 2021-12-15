import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import indexes as ind
from functions import e_matrix_functions as emf
from functions import array_functions as af
from functions import development_functions as df
import copy

af = importlib.reload(af)
df = importlib.reload(df)
emf = importlib.reload(emf)
ind = importlib.reload(ind)

# %% 3um x 140nm
Lx = 3e+3
Ly = 1e+3
D0 = 140

x_min, x_max = -Lx / 2, Lx / 2
z_min, z_max = 0, D0

bin_size = 10

x_bins = np.arange(x_min, x_max + 1, bin_size)
z_bins = np.arange(z_min, z_max + 1, bin_size)

x_centers = (x_bins[:-1] + x_bins[1:]) / 2
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

# atoda: Q_l = 2e-9  # C / cm
# Q_l = 6.56e-9  # C / cm
Q = 4.5e-9 * 223  # A * s = C
Q_line = Q / 625  # C
line_len = Lx * 1e-7 * 625 * 1.3
Q_l = Q_line / line_len

dose_factor = 3.2
beam_sigma = 400

n_electrons_required = Q_l * (Ly * 1e-7) / 1.6e-19
n_files_required = int(n_electrons_required / 100 * dose_factor)

E_dep_array_1_file = np.load('notebooks/development/E_dep_array_1_file.npy')

E_dep_matrix_1_file = np.histogramdd(
    sample=E_dep_array_1_file[:, :2],
    bins=[x_bins, z_bins],
    weights=E_dep_array_1_file[:, 2]
)[0]

# %
E_dep_matrix = np.zeros((len(x_centers), len(z_centers)))

grid_matrix = np.zeros((n_files_required, len(x_centers)))

for i in range(n_files_required):
    grid_matrix[i, :] = x_centers


x_shifts = np.random.normal(
        loc=0,
        scale=beam_sigma,
        size=n_files_required
    )

x_shift_matrix = np.zeros((n_files_required, len(x_centers)))

for i in range(n_files_required):
    x_shift_matrix[i, :] = x_shifts[i]

x_shifts_pos = np.argmin(np.abs(grid_matrix - x_shift_matrix), axis=1)

# hist, bins = np.histogram(a=x_shifts_pos)

# %
# plt.figure(dpi=300)
# plt.plot((bins[1:] + bins[:-1]) / 2, hist)
# plt.show()

# %
E_dep_matrix = np.zeros(np.shape(E_dep_matrix_1_file))

progress_bar = tqdm(total=n_files_required, position=0)

for i in range(n_files_required):

    center_ind = int(len(x_centers) / 2)
    now_shift_ind = x_shifts_pos[i]

    delta_shift_ind = int(now_shift_ind - center_ind)
    # print(delta_shift_ind)
    # delta_shift_ind = -100

    if delta_shift_ind >= 0:
        E_dep_matrix[:delta_shift_ind, :] += E_dep_matrix_1_file[len(x_centers) - delta_shift_ind:, :]
        E_dep_matrix[delta_shift_ind:, :] += E_dep_matrix_1_file[:len(x_centers) - delta_shift_ind, :]
    else:
        E_dep_matrix[:len(x_centers) - -delta_shift_ind, :] += E_dep_matrix_1_file[-delta_shift_ind:, :]
        E_dep_matrix[len(x_centers) - -delta_shift_ind:, :] += E_dep_matrix_1_file[:-delta_shift_ind, :]

    progress_bar.update()

# %
# plt.figure(dpi=300)
# plt.imshow(np.log(E_dep_matrix_1_file.transpose()))
# plt.imshow(np.log(E_dep_matrix).transpose())
# plt.show()

# %
rho = 1.19  # g / cc
Na = 6.02e+23
# Mn = 2e+5
Mn = 2.7e+5
G = 1.9

eps_matrix = E_dep_matrix / bin_size**2 / Ly / 1e-21

Mf_matrix = Mn / (1 + G / 100 * eps_matrix * Mn / (rho * Na))

# plt.figure(dpi=300)
# plt.imshow(np.log(Mf_matrix).transpose())
# plt.show()

# %
# MIBK
# R0 = 51  # A / min
# alpha = 1.42
# beta = 3.59e+8  # A / min

# MIBK : IPA = 1 : 3
R0 = 0.0  # A / min
# beta = 9.3e+14
beta = 1.046e+16
alpha = 3.86

development_rates_A_min = df.get_development_rates_Mf(
    Mf_matrix=Mf_matrix,
    R0=R0,
    alpha=alpha,
    beta=beta
)

development_rates = development_rates_A_min * 0.1 / 60

# %
bin_heights = np.ones(np.shape(development_rates)) * bin_size
now_t = 0

n_cells_removed = len(x_centers) * len(z_centers)

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

    if i % 100 == 0:
        print(now_t)

    times[i] = now_t
    profiles[i, :, :] = copy.deepcopy(bin_heights)

    progress_bar.update()

    if now_t > 30:
        break

# %%
last_profile = profiles[i, :, :]

inds_x, inds_z = np.where(np.logical_and(last_profile < bin_size, last_profile > 0))

plt.figure(dpi=300)
# plt.imshow(profiles[i, :, :].transpose())
plt.plot(x_centers[inds_x], 140 - z_centers[inds_z], '-o')

plt.ylim(0, 160)

plt.grid()

plt.show()
# plt.savefig('30s_profile_x4_500_10nm_bin.jpg', dpi=300)
