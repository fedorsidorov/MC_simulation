import importlib

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import constants as const
import indexes as ind
from mapping import mapping_3p3um_80nm as mapping
from functions import array_functions as af
from functions import e_matrix_functions as emf
from functions import plot_functions as pf
from functions import scission_functions as sf

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
pf = importlib.reload(pf)
sf = importlib.reload(sf)

# %%
lx = 5e+3
ly = 10e+3
lz = 6e+3

r_beam = 100

n_files = 100
n_primaries_in_file = 100
file_cnt = 0

bin_size = 5
z_bins = np.arange(0, lz + 1, bin_size)

x_ind, y_ind, z_ind, E_dep_ind = 4, 5, 6, 7
z_centers = z_bins[:-1] + z_bins[1:]

E_dep_matrix = np.zeros(len(z_centers))

progress_bar = tqdm(total=n_files, position=0)

source = '/Volumes/Transcend/NEW_e_DATA_900nm/'
# source = '/Volumes/Transcend/NEW_e_DATA_900nm_no_interface/'

for _ in range(n_files):

    now_DATA = np.load(source + 'e_DATA_' + str(file_cnt % n_files) + '.npy')
    file_cnt += 1

    now_DATA = now_DATA[np.where(now_DATA[:, E_dep_ind] > 0)]

    E_dep_matrix += np.histogram(now_DATA[:, z_ind], bins=z_bins, weights=now_DATA[:, E_dep_ind])[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(z_centers, E_dep_matrix)
plt.show()

# %%
# np.save('notebooks/heating/heating_5000_files.npy', E_dep_matrix)
# E_dep_matrix = np.load('notebooks/heating/heating_1000_files.npy')

# %%
n_files_required = 1000

ly_m = ly * 1e-9
bin_size_m = bin_size * 1e-9

E_dep_1e = E_dep_matrix / n_files_required / n_primaries_in_file
E_dep_1e_J_m3 = E_dep_1e * 1.6e-19 / (bin_size_m**2 * ly_m)


# %%
def get_heat_source(x, z):

    x_pos = np.argmin(np.abs(x_bins - x))
    z_pos = np.argmin(np.abs(z_bins - z))
    return E_dep_1e_J_m3[x_pos, z_pos] * n_e_1s


# %%
j = 0.85e-9
n_e_1s_cm2 = j / 1.6e-19

n_e_1s = n_e_1s_cm2 * (lx * 1e-7) * (ly * 1e-7)




