import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from functions import plot_functions as pf
from mapping import mapping_harris as mm
import importlib
from matplotlib import rc

pf = importlib.reload(pf)
mm = importlib.reload(mm)

font = {'size': 14}
matplotlib.rc('font', **font)

chain_list = []

# for n in range(754):
for n in range(1447):
    chain_list.append(
        # np.load('data/chains/harris/shifted_snaked_chains/shifted_snaked_chain_' + str(n) + '.npy'))
        np.load('/Volumes/TOSHIBA EXT/chains_Harris/_outdated/shifted_snaked_chains'
                '/shifted_snaked_chain_' + str(n) + '.npy'))

# %%
# font_size = 8
# font_size = 16

# fig = plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=300, figsize=[4, 3])

# fig = plt.figure(dpi=600, figsize=[6.4 / 1.9, 4.8 / 1.9])
# fig = plt.figure(dpi=600, figsize=[4.8 / 1.9, 4.8 / 1.9])
# ax = fig.gca(projection='3d')

# fig = plt.gcf()
# fig.set_size_inches(3, 3)
# fig.set_size_inches(5, 5)
# fig.set_size_inches(5.5, 5.5)

step = 1

# for chain in chain_list[4::45]:
# for chain in chain_list[4::15]:
for chain in chain_list[4::1]:

    plt.plot(chain[::step, 0], chain[::step, 2], '.', markersize=0.5)

# ax.set_xlabel(r'x, nm', fontsize=font_size)
# ax.set_ylabel(r'y, nm', fontsize=font_size)
# ax.set_zlabel(r'z, nm', fontsize=font_size)

# plt.xlabel(r'x, нм', fontsize=18)
# plt.ylabel(r'z, нм', fontsize=18)
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

# plt.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_min,
#         np.ones(mm.l_x) * mm.z_min, 'k')
# plt.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_min,
#         np.ones(mm.l_x) * mm.z_max, 'k')
# ax.plot(np.linspace(mm.x_min, mm.x_max, mm.l_x), np.ones(mm.l_x) * mm.y_max,
#         np.ones(mm.l_x) * mm.z_max, 'k')

# ax.plot(np.ones(mm.l_y) * mm.x_min, np.linspace(mm.y_min, mm.y_max, mm.l_y),
#         np.ones(mm.l_y) * mm.z_min, 'k')
# ax.plot(np.ones(mm.l_y) * mm.x_max, np.linspace(mm.y_min, mm.y_max, mm.l_y),
#         np.ones(mm.l_y) * mm.z_min, 'k')
# ax.plot(np.ones(mm.l_y) * mm.x_min, np.linspace(mm.y_min, mm.y_max, mm.l_y),
#         np.ones(mm.l_y) * mm.z_max, 'k')
# ax.plot(np.ones(mm.l_y) * mm.x_max, np.linspace(mm.y_min, mm.y_max, mm.l_y),
#         np.ones(mm.l_y) * mm.z_max, 'k')

# ax.plot(np.ones(mm.l_z) * mm.x_min, np.ones(mm.l_z) * mm.y_min,
#         np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')
# ax.plot(np.ones(mm.l_z) * mm.x_max, np.ones(mm.l_z) * mm.y_min,
#         np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')
# ax.plot(np.ones(mm.l_z) * mm.x_min, np.ones(mm.l_z) * mm.y_max,
#         np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')
# ax.plot(np.ones(mm.l_z) * mm.x_max, np.ones(mm.l_z) * mm.y_max,
#         np.linspace(mm.z_min, mm.z_max, mm.l_z), 'k')

# plt.xlim(mm.x_min, mm.x_max)
# plt.ylim(mm.y_min, mm.y_max)
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

plt.xlim(-50, 50)
plt.ylim(0, 100)
plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('Harris_chains_1step.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('Harris_chains_2step.jpg', dpi=600, bbox_inches='tight')
plt.savefig('Harris_chains_3step.jpg', dpi=600, bbox_inches='tight')
plt.show()
