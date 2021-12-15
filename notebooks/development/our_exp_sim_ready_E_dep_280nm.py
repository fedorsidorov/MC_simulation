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

# %% 1um x 1um
Lx = 6e+3
Ly = 1e+3
D0 = 280

x_min, x_max = -Lx / 2, Lx / 2
z_min, z_max = 0, D0

bin_size = 10

x_bins = np.arange(x_min, x_max + 1, bin_size)
z_bins = np.arange(z_min, z_max + 1, bin_size)

x_centers = (x_bins[:-1] + x_bins[1:]) / 2
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

beam_sigma = 700

dose_factor = 7

E_dep_matrix = np.load(
    'notebooks/development/E_dep_280nm_10nm/E_dep_280nm_normal_' + str(beam_sigma) + '.npy'
    ) * dose_factor

print(np.sum(E_dep_matrix))

# %
# plt.figure(dpi=300)
# plt.imshow(E_dep_matrix.transpose())
# plt.imshow(np.log(E_dep_matrix).transpose())
# plt.show()

# %%
rho = 1.19  # g / cc
Na = 6.02e+23
# Mn = 2e+5
Mn = 2.7e+5
G = 1.9

eps_matrix = E_dep_matrix / bin_size**2 / Ly / 1e-21

Mf_matrix = Mn / (1 + G / 100 * eps_matrix * Mn / (rho * Na))

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
xx = np.load('notebooks/DEBER_profiles/wet_slice/xx.npy')
zz = np.load('notebooks/DEBER_profiles/wet_slice/zz.npy')

last_profile = profiles[i, :, :]

inds_x, inds_z = np.where(np.logical_and(last_profile < bin_size, last_profile > 0))

plt.figure(dpi=300)
plt.plot(x_centers[inds_x] / 1000, 280 - z_centers[inds_z], label='simulation')

plt.plot(xx, zz, label='slice profile')

plt.xlim(-4, 4)
plt.ylim(0, 300)

plt.title('dose: x' + str(dose_factor) + ', sigma: ' + str(beam_sigma))
plt.xlabel('x, um')
plt.ylabel('z, um')

plt.grid()
plt.legend()

plt.savefig('figures/slice_sim_10nm_normal/dose_x' + str(dose_factor) + '_sigma_' + str(beam_sigma) + '.jpg', dpi=300)
plt.show()
