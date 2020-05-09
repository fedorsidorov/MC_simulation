import importlib
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import mapping_aktary as mapping
from mpl_toolkits.mplot3d import Axes3D
import constants as const
mapping = importlib.reload(mapping)

# %%
folder_name = 'Aktary'
deg_path = 'series_5'

n_surface_facets = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_surface_facets.npy'
)
resist_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/' + folder_name + '/resist_matrix.npy'
)
chain_lens_array = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/' + folder_name + '/prepared_chains/prepared_chain_lens.npy'
)
n_chains = len(chain_lens_array)

chain_tables = []
chains = []

progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chains.append(
        np.load(
            '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/shifted_snaked_chains/shifted_snaked_chain_'
            + str(n) + '.npy')
    )
    chain_tables.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/chains/'
                + folder_name + '/chain_tables_final_series_5/chain_table_' + str(n) + '.npy')
    )
    progress_bar.update()

resist_shape = mapping.hist_2nm_shape

# %%
monomers_deque = deque()

for i in range(mapping.hist_2nm_shape[0]):
    for j in range(mapping.hist_2nm_shape[1]):
        for k in range(mapping.hist_2nm_shape[2]):

            if n_surface_facets[i, k] == 0:
                continue

            mon_lines = resist_matrix[i, j, k]

            for line in mon_lines:
                if line[0] == const.uint32_max:
                    break
                monomers_deque.append(line)

# %%
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

monomers_list = list(monomers_deque)
points_array = np.zeros((len(monomers_list), 3))

for n, mon_line in enumerate(monomers_list):
    n_chain, monomer_pos = mon_line[0], mon_line[1]
    mon_x, mon_y, mon_z = chains[n_chain][monomer_pos]
    points_array[n, :] = mon_x, mon_y, mon_z

plt.plot(points_array[::5, 0], points_array[::5, 2], '.')
plt.show()


