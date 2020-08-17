# %% Import
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %% read datafile
l_x = 10
l_y = 10
nx = 20 + 1
ny = 20 + 1

xx = np.linspace(-l_x / 2, l_x / 2, nx)
yy = np.linspace(-l_y / 2, l_y / 2, ny)
zz = np.zeros((len(xx), len(yy)))

for i in range(len(xx)):
    for j in range(len(yy)):
        zz[i, j] = np.sin(xx[i] * yy[j]) + 1

plt.figure(dpi=300)
plt.imshow(zz)
plt.show()

# %% vertices
n_vertices = nx * ny * 2
VV = np.zeros((n_vertices, 1 + 3))
VV_inv = np.zeros((2, nx, ny), dtype=int)
cnt = 0

for i, x in enumerate(xx):
    for j, y in enumerate(yy):
        VV[cnt, :] = cnt, x, y, 0
        VV[nx * ny + cnt, :] = nx * ny + cnt, x, y, zz[i, j]
        VV_inv[0, i, j] = cnt
        VV_inv[1, i, j] = nx * ny + cnt
        cnt += 1

# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(VV[:, 1], VV[:, 2], VV[:, 3], 'o')
# plt.show()

# %% edges
n_edges = ((nx - 1) * ny + (ny - 1) * nx) * 2 + nx * 2 + ny * 2 - 4
EE = np.zeros((n_edges, 1 + 2))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')  # sdf
cnt = 0

for i in range(nx-1):  # lower layer ^
    for j in range(ny):
        EE[cnt, :] = cnt, VV_inv[0, i, j], VV_inv[0, i+1, j]
        cnt += 1

for i in range(nx):  # lower layer >
    for j in range(ny-1):
        EE[cnt, :] = cnt, VV_inv[0, i, j], VV_inv[0, i, j+1]
        cnt += 1

for i in range(nx-1):  # upper layer ^
    for j in range(ny):
        EE[cnt, :] = cnt, VV_inv[1, i, j], VV_inv[1, i+1, j]
        cnt += 1

for i in range(nx):  # upper layer >
    for j in range(ny-1):
        EE[cnt, :] = cnt, VV_inv[1, i, j], VV_inv[1, i, j+1]
        cnt += 1

for i in range(nx):
    EE[cnt, :] = cnt, VV_inv[0, i, 0], VV_inv[1, i, 0]
    cnt += 1

for i in range(nx):
    EE[cnt, :] = cnt, VV_inv[0, i, ny-1], VV_inv[1, i, ny-1]
    cnt += 1

for j in range(1, ny-1):
    EE[cnt, :] = cnt, VV_inv[0, 0, j], VV_inv[1, 0, j]
    cnt += 1

for j in range(1, ny-1):
    EE[cnt, :] = cnt, VV_inv[0, nx-1, j], VV_inv[1, nx-1, j]
    cnt += 1

