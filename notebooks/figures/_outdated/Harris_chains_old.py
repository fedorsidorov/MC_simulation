import matplotlib.pyplot as plt
import numpy as np
from functions import plot_functions as pf
from mapping import mapping_harris as mm
import importlib
from matplotlib import rc

pf = importlib.reload(pf)
mm = importlib.reload(mm)


chain_list = []

# for n in range(754):
for n in range(1447):
    chain_list.append(
        # np.load('data/chains/harris/shifted_snaked_chains/shifted_snaked_chain_' + str(n) + '.npy'))
        np.load('/Volumes/TOSHIBA EXT/chains_Harris/_outdated/shifted_snaked_chains'
                '/shifted_snaked_chain_' + str(n) + '.npy'))

# %%
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Helvetica"]})

# font_size = 8
# font_size = 16

# fig = plt.figure(dpi=600)
# fig = plt.figure(dpi=600, figsize=[6.4 / 1.9, 4.8 / 1.9])
fig = plt.figure(dpi=600, figsize=[4.8 / 1.9, 4.8 / 1.9])
ax = fig.gca(projection='3d')

fig = plt.gcf()
# fig.set_size_inches(3, 3)
# fig.set_size_inches(5.5, 5.5)

step = 1

for chain in chain_list[4::45]:

    # if len(chain) > 100000:
    #     continue

    # ax.plot(chain[::step, 0], chain[::step, 1], chain[::step, 2], '.', markersize=1)
    ax.plot(chain[::step, 0], chain[::step, 1], chain[::step, 2], '.', markersize=0.5)

# ax.set_xlabel(r'x, nm', fontsize=font_size)
# ax.set_ylabel(r'y, nm', fontsize=font_size)
# ax.set_zlabel(r'z, nm', fontsize=font_size)

ax.set_xlabel(r'x, nm')
ax.set_ylabel(r'y, nm')
ax.set_zlabel(r'z, nm')

ax.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_min,
        np.ones(mm.l_x) * mm.z_min, 'k')
# ax.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_max,
#         np.ones(mm.l_x) * mm.z_min, 'k')
ax.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_min,
        np.ones(mm.l_x) * mm.z_max, 'k')
ax.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_max,
        np.ones(mm.l_x) * mm.z_max, 'k')

# ax.plot(np.ones(mm.l_y) * mm.x_min, np.linspace(mm.y_min, mm.y_max, mm.l_y),
#         np.ones(mm.l_y) * mm.z_min, 'k')
ax.plot(np.ones(mm.l_y) * mm.x_max, np.linspace(mm.y_min, mm.y_max, mm.l_y),
        np.ones(mm.l_y) * mm.z_min, 'k')
ax.plot(np.ones(mm.l_y) * mm.x_min, np.linspace(mm.y_min, mm.y_max, mm.l_y),
        np.ones(mm.l_y) * mm.z_max, 'k')
ax.plot(np.ones(mm.l_y) * mm.x_max, np.linspace(mm.y_min, mm.y_max, mm.l_y),
        np.ones(mm.l_y) * mm.z_max, 'k')

ax.plot(np.ones(mm.l_z) * mm.x_min, np.ones(mm.l_z) * mm.y_min,
        np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')
ax.plot(np.ones(mm.l_z) * mm.x_max, np.ones(mm.l_z) * mm.y_min,
        np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')
# ax.plot(np.ones(mm.l_z) * mm.x_min, np.ones(mm.l_z) * mm.y_max,
#         np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')
ax.plot(np.ones(mm.l_z) * mm.x_max, np.ones(mm.l_z) * mm.y_max,
        np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')

plt.xlim(mm.x_min, mm.x_max)
plt.ylim(mm.y_min, mm.y_max)
# plt.title('Polymer chain simulation')

# ax.set_xlabel('$x$, nm', fontsize=font_size)
# ax.set_ylabel('$y$, nm', fontsize=font_size)
# ax.set_zlabel('$z$, nm', fontsize=font_size)

# ax.set_xlabel('$x$, нм', fontsize=font_size)
# ax.set_ylabel('$y$, нм', fontsize=font_size)
# ax.set_zlabel('$z$, нм', fontsize=font_size)

# ax = plt.gca()
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)
# for tick in ax.zaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)

plt.savefig('figures/Harris_chains.jpg', dpi=600, bbox_inches='tight')
plt.show()
