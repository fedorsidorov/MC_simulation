import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import indexes as ind
from functions import e_matrix_functions as emf
from functions import array_functions as af
from functions import development_functions_2d as df

af = importlib.reload(af)
df = importlib.reload(df)
emf = importlib.reload(emf)
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
beam_sigma = (2 / Rb)**2

n_electrons_required = Q_l * Ly * 1e-7 / 1.6e-19
n_files_required = int(n_electrons_required / 100)

E_dep_matrix = np.zeros((len(x_centers), len(z_centers)))

# %%
progress_bar = tqdm(total=n_files_required, position=0)

n_files = 117
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

    af.snake_coord(
        array=now_e_DATA,
        coord_ind=4,
        coord_min=x_min,
        coord_max=x_max
    )

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
E_dep_matrix = np.load('notebooks/development/E_dep_atoda.npy') * 4.25


# %%
plt.figure(dpi=300)
# plt.imshow(E_dep_matrix.transpose())
plt.imshow(np.log(E_dep_matrix).transpose())
plt.show()

# %%
rho = 1.19  # g / cc
Na = 6.02e+23

# eps_matrix = E_dep_matrix / 1e-7**3
eps_matrix = E_dep_matrix / bin_size**2 / Ly / 1e-21

Mf_matrix = Mn / (1 + G / 100 * eps_matrix * Mn / (rho * Na))

plt.figure(dpi=300)
plt.imshow(np.log(Mf_matrix).transpose())
plt.show()

# %%
# MIBK
# R0 = 51  # A / min
# alpha = 1.42
# beta = 3.59e+8  # A / min

# MIBK : IPA = 1 : 3
R0 = 0.0  # A / min
beta = 9.3e+14
alpha = 3.86

development_rates = df.get_development_rates_Mf(
    Mf_matrix=Mf_matrix,
    R0=R0,
    alpha=alpha,
    beta=beta
)

development_rates_nm_s = development_rates * 0.1 / 60

bin_size = 10  # nm

development_times = bin_size / development_rates_nm_s
n_surface_facets = df.get_initial_n_surface_facets(Mf_matrix)

# %%
n_seconds = 30
factor = 5

delta_t = 1 / factor
n_steps = n_seconds * factor

progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):
    df.make_develop_step(development_times, n_surface_facets, delta_t)
    progress_bar.update()

progress_bar.close()

# %%
plt.figure(dpi=300)
# plt.imshow(n_surface_facets.transpose())
plt.imshow(np.log(development_times).transpose())
plt.colorbar()
plt.show()


