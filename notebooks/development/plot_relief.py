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
deg_path = 'series_2'

n_surface_facets = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/development/n_surface_facets_4nm_15s.npy'
)
resist_matrix = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/chains/' + folder_name + '/best_resist_matrix_4nm.npy'
)
n_chains = 754

chain_tables = []
chains = []

progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chains.append(
        np.load(
            '/Users/fedor/PycharmProjects/MC_simulation/data/chains/Aktary/best_sh_sn_chains/sh_sn_chain_'
            + str(n) + '.npy')
    )
    chain_tables.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/chains/'
                + folder_name + '/best_chain_tables_series_2_4nm/chain_table_' + str(n) + '.npy')
    )
    progress_bar.update()

progress_bar.close()

resist_shape = mapping.hist_4nm_shape

# %%
monomers_deque = deque()

for i in range(mapping.hist_4nm_shape[0]):
    for j in range(mapping.hist_4nm_shape[1]):
        for k in range(mapping.hist_4nm_shape[2]):

            if n_surface_facets[i, j, k] == 0:
                continue

            mon_lines = resist_matrix[i, j, k]

            for line in mon_lines:
                if line[0] == const.uint32_max:
                    break
                monomers_deque.append(line)

# %%
monomers_list = list(monomers_deque)
points_array = np.zeros((len(monomers_list), 3))

for n, mon_line in enumerate(monomers_list):
    n_chain, monomer_pos = mon_line[0], mon_line[1]
    mon_x, mon_y, mon_z = chains[n_chain][monomer_pos]
    points_array[n, :] = mon_x, mon_y, mon_z

xx = points_array[:, 0]
yy = points_array[:, 1]
zz = points_array[:, 2]

# %%
np.save('data/chains/Aktary/development/xx.npy', xx)
np.save('data/chains/Aktary/development/yy.npy', yy)
np.save('data/chains/Aktary/development/zz.npy', zz)

# %%
fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(xx, yy, zz, '.')
plt.plot(xx, yy, '.')
# plt.plot(xx, yy)
plt.show()
