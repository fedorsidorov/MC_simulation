import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from functions import plot_functions as pf
import mapping_harris as const_m
# import mapping_aktary as const_m
import importlib
pf = importlib.reload(pf)
const_m = importlib.reload(const_m)


chain_list = []

# for n in range(754):
for n in range(1447):
    chain_list.append(
        # np.load('data/chains/Harris/shifted_snaked_chains/shifted_snaked_chain_' + str(n) + '.npy'))
        np.load('/Volumes/ELEMENTS/chains_Harris/_outdated/shifted_snaked_chains'
                '/shifted_snaked_chain_' + str(n) + '.npy'))

# %%
# font_size = 8
font_size = 14

fig = plt.figure(dpi=300)
ax = fig.gca(projection='3d')

fig = plt.gcf()
# fig.set_size_inches(3, 3)
fig.set_size_inches(5.5, 5.5)

step = 1

for chain in chain_list[4::45]:

    # if len(chain) > 100000:
    #     continue

    ax.plot(chain[::step, 0], chain[::step, 1], chain[::step, 2], 'o', markersize=1)

ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_min,
        np.ones(const_m.l_x)*const_m.z_min, 'k')
ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_max,
        np.ones(const_m.l_x)*const_m.z_min, 'k')
ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_min,
        np.ones(const_m.l_x)*const_m.z_max, 'k')
ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_max,
        np.ones(const_m.l_x)*const_m.z_max, 'k')

ax.plot(np.ones(const_m.l_y)*const_m.x_min, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
        np.ones(const_m.l_y)*const_m.z_min, 'k')
ax.plot(np.ones(const_m.l_y)*const_m.x_max, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
        np.ones(const_m.l_y)*const_m.z_min, 'k')
ax.plot(np.ones(const_m.l_y)*const_m.x_min, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
        np.ones(const_m.l_y)*const_m.z_max, 'k')
ax.plot(np.ones(const_m.l_y)*const_m.x_max, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
        np.ones(const_m.l_y)*const_m.z_max, 'k')

ax.plot(np.ones(const_m.l_z)*const_m.x_min, np.ones(const_m.l_z)*const_m.y_min,
        np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')
ax.plot(np.ones(const_m.l_z)*const_m.x_max, np.ones(const_m.l_z)*const_m.y_min,
        np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')
ax.plot(np.ones(const_m.l_z)*const_m.x_min, np.ones(const_m.l_z)*const_m.y_max,
        np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')
ax.plot(np.ones(const_m.l_z)*const_m.x_max, np.ones(const_m.l_z)*const_m.y_max,
        np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')

plt.xlim(const_m.x_min, const_m.x_max)
plt.ylim(const_m.y_min, const_m.y_max)
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

# %%
plt.savefig('chains.tiff', bbox_inches='tight')
