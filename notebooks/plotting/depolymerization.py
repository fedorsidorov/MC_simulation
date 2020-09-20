import matplotlib.pyplot as plt
import numpy as np
from functions import plot_functions as pf
from mapping import mapping_harris as const_m
# import mapping_aktary as const_m
import importlib
pf = importlib.reload(pf)
const_m = importlib.reload(const_m)


chain_list = []

# for n in range(754):
for n in range(1447):
    chain_list.append(
        # np.load('data/chains/harris/shifted_snaked_chains/shifted_snaked_chain_' + str(n) + '.npy'))
        np.load('/Volumes/ELEMENTS/chains_Harris/_outdated/shifted_snaked_chains'
                '/shifted_snaked_chain_' + str(n) + '.npy'))

# %%
# font_size = 8
font_size = 14

fig = plt.figure(dpi=600)
ax = fig.gca(projection='3d')

fig = plt.gcf()
# fig.set_size_inches(3, 3)
fig.set_size_inches(5.5, 5.5)

step = 1

n_chain = 200

# for chain in chain_list[n_chain:n_chain + 1]:
chain = chain_list[n_chain]
ax.plot(chain[::step, 0], chain[::step, 1], chain[::step, 2], 'o', markersize=1)

ax.plot(chain[3000:5000:step, 0], chain[3000:5000:step, 1], chain[3000:5000:step, 2], 'o',
        color='orange', markersize=1)
ax.plot(chain[3000:3001, 0], chain[3000:3001, 1], chain[3000:3001, 2], 'ro', markersize=8)

ax.plot(chain[10000:12000:step, 0], chain[10000:12000:step, 1], chain[10000:12000:step, 2], 'o',
        color='orange', markersize=1)
ax.plot(chain[10000:10001, 0], chain[10000:10001, 1], chain[10000:10001, 2], 'ro', markersize=8)

ax.plot(chain[18000:20000:step, 0], chain[18000:20000:step, 1], chain[18000:20000:step, 2], 'o',
        color='orange', markersize=1)
ax.plot(chain[18000:18001, 0], chain[18000:18001, 1], chain[18000:18001, 2], 'ro', markersize=8)

eps = 0.01
ax.axes.set_xlim3d(left=-10+eps, right=40-eps)
ax.axes.set_ylim3d(bottom=-30+eps, top=20-eps)
ax.axes.set_zlim3d(bottom=180+eps, top=230-eps)

# plt.title('Polymer chain simulation')

# ax.set_xlabel('$x$, nm', fontsize=font_size)
# ax.set_ylabel('$y$, nm', fontsize=font_size)
# ax.set_zlabel('$z$, nm', fontsize=font_size)

ax.set_xlabel('$x$, нм', fontsize=font_size)
ax.set_ylabel('$y$, нм', fontsize=font_size)
ax.set_zlabel('$z$, нм', fontsize=font_size)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

# plt.show()

# %
plt.savefig('depolymerization.tiff', bbox_inches='tight')
