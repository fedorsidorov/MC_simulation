import importlib
import os
import matplotlib.pyplot as plt
import numpy as np
from functions import MC_functions as mf

mf = importlib.reload(mf)

# %% read datafile
h0 = 80  # nm
lx = 50  # nm
ly = 200  # nm

yy_pre = np.linspace(-ly/40, ly/40, 20)
zz_pre = h0 * (1 - np.cos(yy_pre / ly * 20 * 2 * np.pi)) / 4 + 40

yy_beg = np.linspace(-ly/2, -ly/40 - 1, 20)
yy_end = np.linspace(ly/40 + 1, ly/2, 20)
zz_beg = np.ones(len(yy_beg)) * zz_pre[0]
zz_end = np.ones(len(yy_end)) * zz_pre[-1]

yy_pre = np.concatenate((yy_beg, yy_pre, yy_end))
zz_pre = np.concatenate((zz_beg, zz_pre, zz_end))

z_max = np.max(zz_pre)

plt.figure(dpi=300)
plt.plot(yy_pre, zz_pre, '.-')
plt.show()

# %%
ny = len(yy_pre)
nx = 2

xx = np.linspace(-lx/2, lx/2, nx)
yy = yy_pre
zz = np.zeros((nx, ny))

for i in range(len(xx)):
    zz[i, :] = zz_pre

volume = 0

for i in range(len(xx) - 1):
    volume += np.trapz(zz[i, :], x=yy) * (xx[1] - xx[0])

# %%
# plt.figure(dpi=300)
# plt.plot(yy, zz[0, :], '.-')
# plt.show()

# %%
# % vertices
n_vertices = nx * ny * 2
VV = np.zeros((n_vertices, 1 + 3))
VV_inv = np.zeros((2, nx, ny), dtype=int)
cnt = 1

for i, x in enumerate(xx):
    for j, y in enumerate(yy):
        VV[cnt - 1, :] = cnt, x, y, 0
        VV[nx * ny + cnt - 1, :] = nx * ny + cnt, x, y, zz[i, j]
        VV_inv[0, i, j] = cnt
        VV_inv[1, i, j] = nx * ny + cnt
        cnt += 1

# % edges
n_edges = ((nx - 1) * ny + (ny - 1) * nx) * 2 + nx * 2 + ny * 2 - 4
EE = np.zeros((n_edges, 1 + 2), dtype=int)
VV_EE = np.zeros((2, nx, ny, 6), dtype=int)

# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111, projection='3d')  # sdf
cnt = 1

for i in range(nx - 1):  # lower layer >
    for j in range(ny):
        EE[cnt - 1, :] = cnt, VV_inv[0, i, j], VV_inv[0, i + 1, j]
        VV_EE[0, i, j, 0] = cnt
        VV_EE[0, i + 1, j, 2] = cnt
        cnt += 1

for i in range(nx):  # lower layer ^
    for j in range(ny - 1):
        EE[cnt - 1, :] = cnt, VV_inv[0, i, j], VV_inv[0, i, j + 1]
        VV_EE[0, i, j, 1] = cnt
        VV_EE[0, i, j + 1, 3] = cnt
        cnt += 1

for i in range(nx - 1):  # lower layer >
    for j in range(ny):
        EE[cnt - 1, :] = cnt, VV_inv[1, i, j], VV_inv[1, i + 1, j]
        VV_EE[1, i, j, 0] = cnt
        VV_EE[1, i + 1, j, 2] = cnt
        cnt += 1

for i in range(nx):  # lower layer ^
    for j in range(ny - 1):
        EE[cnt-1, :] = cnt, VV_inv[1, i, j], VV_inv[1, i, j + 1]
        VV_EE[1, i, j, 1] = cnt
        VV_EE[1, i, j + 1, 3] = cnt
        cnt += 1

for i in range(nx):  # lower-upper
    for j in range(ny):
        if not ((i == 0 or j == 0) or (i == nx - 1 or j == ny - 1)):
            continue
        EE[cnt - 1, :] = cnt, VV_inv[0, i, j], VV_inv[1, i, j]
        VV_EE[0, i, j, 4] = cnt
        VV_EE[1, i, j, 5] = cnt
        cnt += 1

# % faces
n_faces = (nx - 1) * (ny - 1) * 2 + (nx - 1) * 2 + (ny - 1) * 2
FF = np.zeros((n_faces, 1 + 4), dtype=int)

cnt = 1

for i in range(nx - 1):  # bottom
    for j in range(ny - 1):
        FF[cnt-1, :] = cnt, VV_EE[0, i, j, 1], VV_EE[0, i, j + 1, 0],\
                       -VV_EE[0, i + 1, j, 1], -VV_EE[0, i, j, 0]
        cnt += 1

for i in range(nx - 1):  # top
    for j in range(ny - 1):
        FF[cnt-1, :] = cnt, VV_EE[1, i, j, 0], VV_EE[1, i + 1, j, 1],\
                       -VV_EE[1, i, j + 1, 0], -VV_EE[1, i, j, 1]
        cnt += 1

for i in range(nx - 1):  # front
    FF[cnt - 1, :] = cnt, VV_EE[0, i, 0, 0], VV_EE[0, i + 1, 0, 4],\
                     -VV_EE[1, i + 1, 0, 2], -VV_EE[1, i, 0, 5]
    cnt += 1

for i in range(nx - 1):  # back
    FF[cnt - 1, :] = cnt, VV_EE[0, i, ny - 1, 4], VV_EE[1, i, ny - 1, 0], \
                     -VV_EE[1, i + 1, ny - 1, 5], -VV_EE[0, i + 1, ny - 1, 2]
    cnt += 1

for j in range(ny - 1):  # left
    FF[cnt - 1, :] = cnt, VV_EE[0, 0, j, 4], VV_EE[1, 0, j, 1], \
                     -VV_EE[1, 0, j + 1, 5], -VV_EE[0, 0, j + 1, 3]
    cnt += 1

for j in range(ny - 1):  # right
    FF[cnt - 1, :] = cnt, VV_EE[0, nx - 1, j, 1], VV_EE[0, nx - 1, j + 1, 4], \
                     -VV_EE[1, nx - 1, j + 1, 3], -VV_EE[1, nx - 1, j, 5]
    cnt += 1

# % prepare datafile
file = ''
file += 'define vertex attribute vmob real\n\n'

file += 'MOBILITY_TENSOR \n' + \
           'vmob  0     0\n' + \
           '0     vmob  0\n' + \
           '0     0  vmob\n\n'

file += 'PARAMETER lx = ' + str(lx) + '\n'
file += 'PARAMETER ly = ' + str(ly) + '\n'
file += 'PARAMETER z_max = ' + str(z_max) + '\n'
file += 'PARAMETER angle_surface = 55' + '\n'
file += 'PARAMETER angle_mirror = 90' + '\n'

file += 'PARAMETER TENS_r = 33.5e-2' + '\n'
file += 'PARAMETER TENS_s = -TENS_r*cos((angle_surface)*pi/180)' + '\n'
file += 'PARAMETER TENS_m = -TENS_r*cos((angle_mirror)*pi/180)' + '\n\n'

file += '/*--------------------CONSTRAINTS START--------------------*/\n'

file += 'constraint 1\n'
file += 'formula: z = 0\n\n'

file += 'constraint 11\n'
file += 'formula: x = -lx/2\n\n'

file += 'constraint 22\n'
file += 'formula: y = -ly/2\n\n'

file += 'constraint 33\n'
file += 'formula: x = lx/2\n\n'

file += 'constraint 44\n'
file += 'formula: y = ly/2\n\n'

file += 'constraint 5 nonpositive\n'
file += 'formula: z = z_max\n\n'

file += '/*--------------------CONSTRAINTS END--------------------*/\n\n'

# % vertices
file += '/*--------------------VERTICES START--------------------*/\n'
file += 'vertices\n'

for line in VV:
    file += str(int(line[0])) + '\t' + str(line[1:])[1:-1] + '\n'

file += '/*--------------------VERTICES END--------------------*/\n\n'

# % edges
file += '/*--------------------EDGES START--------------------*/\n'
file += 'edges' + '\n'

for line in EE:
    file += str(line[0]) + '\t' + str(line[1:])[1:-1] + ' color red' + '\n'

file += '/*--------------------EDGES END--------------------*/\n\n'

# % faces
file += '/*--------------------FACES START--------------------*/\n'
file += 'faces' + '\n'

for i in range(len(FF)):
    file += str(FF[i, 0]) + '\t' + str(FF[i, 1:])[1:-1] + '\n'

file += '/*--------------------FACES END--------------------*/\n\n'

# % bodies
file += '/*--------------------BODIES START--------------------*/\n'
file += 'bodies' + '\n' + '1' + '\t'

for fn in FF[:, 0]:
    file += str(fn) + ' '

file += '\tvolume ' + str(volume) + '\n'

file += '\n/*--------------------BODIES END--------------------*/\n\n'

with open('notebooks/SE/set_SE_constraints.txt', 'r') as myfile:
    file += myfile.read()

# % write to file
with open('notebooks/SE/SE_input_3D_test_viscosity.fe', 'w') as myfile:
    myfile.write(file)


# %%
os.system('evolver -f/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands.txt ' +
          '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SE_input_3D_test_viscosity.fe')

# %%
vlist = np.loadtxt('notebooks/SE/vlist.txt')

# vlist = vlist[np.where(vlist[:, 2] != -100)]

plt.figure(dpi=300)
plt.plot(vlist[:, 1], vlist[:, 2], '.')
plt.show()
