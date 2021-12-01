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

# %%
pitch = 6e+3
ly, lz = 1e+10, 270

x_min, x_max = -pitch/2, pitch/2
y_min, y_max = -ly/2, ly/2
z_min, z_max = 0, 270

bin_size = 10

x_bins = np.arange(x_min, x_max + 1, bin_size)
y_bins = np.array([y_min, y_max])
z_bins = np.arange(z_min, z_max + 1, bin_size)

x_centers = (x_bins[:-1] + x_bins[1:]) / 2
y_centers = (y_bins[:-1] + y_bins[1:]) / 2
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

# E_dep_matrix = np.zeros((len(x_centers), len(z_centers)))
E_dep_matrix = np.zeros((len(x_centers), len(y_centers), len(z_centers)))

# %%
n_files_required = int(4.1e+5 / 2)

progress_bar = tqdm(total=n_files_required, position=0)

n_files = 100
file_cnt = 0

while file_cnt < n_files_required:

    primary_electrons_in_file = 100

    now_e_DATA = np.load('data/4_WET/e_DATA_Pn_' + str(file_cnt % n_files) + '.npy')
    now_e_DATA = now_e_DATA[np.where(now_e_DATA[:, 7] > 0)]

    if file_cnt > n_files:
        emf.rotate_DATA(now_e_DATA, x_ind=4, y_ind=5)

    emf.add_gaussian_xy_shift_to_e_DATA(
        e_DATA=now_e_DATA,
        x_position=0,
        x_sigma=100,
        y_range=[y_min, y_max])

    # af.snake_array(
    #     array=now_e_DATA,
    #     x_ind=4,
    #     y_ind=5,
    #     z_ind=6,
    #     xyz_min=[x_min, y_min, z_min],
    #     xyz_max=[x_max, y_max, z_max]
    # )

    af.snake_coord(
        array=now_e_DATA,
        coord_ind=4,
        coord_min=x_min,
        coord_max=x_max
    )

    E_dep_matrix += np.histogramdd(
        sample=now_e_DATA[:, ind.e_DATA_coord_inds],
        bins=[x_bins, y_bins, z_bins],
        weights=now_e_DATA[:, 7]
    )[0]

    file_cnt += 1
    progress_bar.update()


# %%
E_dep_matrix_33_0p5 = E_dep_matrix[:, 0, :]

E_dep_matrix_0p5 = E_dep_matrix_33_0p5 * 3

E_dep_matrix_final = E_dep_matrix_0p5 + E_dep_matrix_0p5[::-1, :]

# %%
plt.figure(dpi=300)
# plt.imshow(np.log(E_dep_matrix[:, 0, :]).transpose())
plt.imshow(np.log(E_dep_matrix_final).transpose())
plt.show()

# %%
# e_matrix_E_dep_rough = e_matrix_E_dep[:, 0, :] * 3
E_dep_matrix = np.load('notebooks/development/E_dep_200nm_beam.npy')

plt.figure(dpi=300)
plt.imshow(np.log(E_dep_matrix).transpose())
plt.show()

# %%
Mn = 271374  # g / mole
g = 0.019  # scission / eV
rho = 1.19  # g / cc
Na = 6.02e+23

# eps_matrix = E_dep_matrix / 1e-7**3
eps_matrix = E_dep_matrix / 1e-7**2 / 10e-4

Mf_matrix = Mn / (1 + g * eps_matrix * Mn / (rho * Na))

plt.figure(dpi=300)
plt.imshow(np.log(Mf_matrix).transpose())
plt.show()

# %%
R0 = 0.0  # A / min
beta = 9.3e+14
alpha = 3.86

# atoda1979
# R0 = 51
# alpha = 1.42
# beta = 3.59e+8

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
n_seconds = 1
factor = 10

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

