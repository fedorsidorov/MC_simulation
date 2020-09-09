import importlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from functions import MC_functions as mf

mf = importlib.reload(mf)

# %% read datafile
profile = np.loadtxt('notebooks/Leveder/2010/0.txt')
h0 = 43  # nm

lx = 10
ly = 10

y_step = 2

profile[0, 0] = 0
profile[-1, 0] = 2
profile[0, 1] = profile[-1, 1]

yy_pre = profile[:, 0]
zz_pre = profile[:, 1]

yy_f = np.concatenate((yy_pre - y_step*2.5, yy_pre - y_step*1.5, yy_pre - y_step*0.5,
                       yy_pre + y_step*0.5, yy_pre + y_step*1.5))
zz_f = (np.concatenate((zz_pre, zz_pre, zz_pre, zz_pre, zz_pre)) + h0) * 1e-3

# plt.figure(dpi=300)
# plt.plot(yy_f, zz_f, 'o-')
# plt.show()

# %
# nx = len(yy_f)
# ny = len(yy_f)
nx = 60
ny = 60

xx = np.linspace(-lx/2, lx/2, nx)
yy = np.linspace(-ly/2, ly/2, ny)
# yy = yy_f
zz = np.zeros((len(xx), len(yy)))

for i in range(len(xx)):
    zz[i, :] = mf.lin_lin_interp(yy_f, zz_f)(yy)
    # zz[i, :] = zz_f

volume = 0

for j in range(len(yy)):
    volume += np.trapz(zz[:, j], x=xx) * (yy[1] - yy[0])

# plt.figure(dpi=300)
# plt.plot(yy, zz[0, :], '.-')
# plt.show()

# %
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
    for j in range(ny-1):
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
    file += str(line[0]) + '\t' + str(line[1:])[1:-1] + '\n'

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
with open('notebooks/SE/SE_input_3D_leveder.fe', 'w') as myfile:
    myfile.write(file)
