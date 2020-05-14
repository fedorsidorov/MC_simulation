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
n_surface_facets = np.load('data/chains/Aktary/development/n_surface_facets_series_1_1500_1nm.npy')
resist_matrix = np.load('data/chains/' + folder_name + '/best_resist_matrix_1nm.npy')
n_chains = 754

chains = []
chain_tables = []

progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chains.append(np.load('data/chains/Aktary/best_sh_sn_chains/sh_sn_chain_' + str(n) + '.npy'))
    chain_tables.append(np.load('data/chains/Aktary/best_chain_tables_series_1_1nm_1500/chain_table_' +
                                str(n) + '.npy'))
    progress_bar.update()

progress_bar.close()

resist_shape = mapping.hist_1nm_shape

# %%
plt.imshow(n_surface_facets.transpose())
plt.show()

# %%
shape_deque = deque()
body_deque = deque()

# j = 50
j_range = range(0, 1)

progress_bar = tqdm(total=mapping.hist_1nm_shape[2], position=0)

for k in range(mapping.hist_1nm_shape[2]):
    surf_positions = np.where(n_surface_facets[:, k] > 0)[0]

    for i in range(mapping.hist_1nm_shape[0]):

        if np.min(surf_positions) < i < np.max(surf_positions):
            continue

        for j in j_range:

            mon_lines = resist_matrix[i, j, k]

            if n_surface_facets[i, k] > 0:
                for line in mon_lines:
                    if line[0] == const.uint32_max:
                        break
                    shape_deque.append(line)

            else:
                for line in mon_lines:
                    if line[0] == const.uint32_max:
                        break
                    body_deque.append(line)

    progress_bar.update()

# %
body_array = np.zeros((len(body_deque), 3))
shape_array = np.zeros((len(shape_deque), 3))

for n, mon_line in enumerate(body_deque):
    n_chain, monomer_pos = mon_line[0], mon_line[1]
    mon_x, mon_y, mon_z = chains[n_chain][monomer_pos]
    body_array[n, :] = mon_x, mon_y, mon_z

for n, mon_line in enumerate(shape_deque):
    n_chain, monomer_pos = mon_line[0], mon_line[1]
    mon_x, mon_y, mon_z = chains[n_chain][monomer_pos]
    shape_array[n, :] = mon_x, mon_y, mon_z

# %
y_min = -50
y_max = 50

inds = np.where(np.logical_and(
    np.logical_and(shape_array[:, 1] >= y_min, shape_array[:, 1] < y_max),
    np.abs(shape_array[:, 0]) < 40)
)[0]

font_size = 8
_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
fig.set_size_inches(3, 3)

plt.plot(body_array[:, 0], 100 - body_array[:, 2], 'ko', markersize=2)  # was 11
plt.plot(shape_array[inds, 0], 100 - shape_array[inds, 2], 'ko', markersize=2)

plt.xlim(-50, 50)
plt.ylim(0, 100)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.xlabel(r'$x$, nm', fontsize=font_size)
plt.ylabel(r'$z$, nm', fontsize=font_size)
plt.gca().set_aspect('equal', adjustable='box')

# plt.show()

# %%
# plt.savefig('final_1nm.tiff', bbox_inches='tight')
plt.savefig('final_1nm.png', bbox_inches='tight')
