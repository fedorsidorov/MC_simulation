import importlib
import os
import numpy as np
from mapping import mapping_3p3um_80nm as mm
import matplotlib.pyplot as plt
from functions import MC_functions as mcf
from scipy.optimize import curve_fit

mm = importlib.reload(mm)
mcf = importlib.reload(mcf)


# %%
def create_datafile_old(yy_pre_pre, zz_pre_pre, mobs_pre_pre):  # xx and zz in um !!!

    l0 = (mm.x_max - mm.x_min) * 1e-3
    n_periods_y = 4

    yy_list = []
    zz_list = []
    mob_list = []

    for n in range(-int(n_periods_y / 2), int(n_periods_y / 2) + 1):
        yy_list.append(yy_pre_pre + l0 * n)
        zz_list.append(zz_pre_pre)
        mob_list.append(mobs_pre_pre)

    yy_pre = np.concatenate(yy_list)
    zz_pre = np.concatenate(zz_list)
    mobs_pre = np.concatenate(mob_list)

    yy = np.linspace(yy_pre.min(), yy_pre.max(), 100)
    zz_1d = mcf.lin_lin_interp(yy_pre, zz_pre)(yy)
    mobs = mcf.lin_log_interp(yy_pre, mobs_pre)(yy)

    nx = ny = len(yy)
    lx = ly = yy.max() - yy.min()

    xx = np.linspace(-ly / 2, ly / 2, nx)
    zz = np.zeros((nx, ny))

    for i in range(len(xx)):
        zz[i, :] = zz_1d

    volume = 0

    # plt.figure(dpi=300)
    # plt.plot(yy, zz[0, :], '.-')
    # plt.show()

    for i in range(len(xx)):
        volume += np.trapz(zz[i, :], x=yy) * (xx[1] - xx[0])

    # vertices
    n_vertices = nx * ny * 2
    VV = np.zeros((n_vertices, 1 + 3 + 1))
    VV_inv = np.zeros((2, nx, ny), dtype=int)
    cnt = 1

    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            VV[cnt - 1, :] = cnt, x, y, 0, mobs[j]
            VV[nx * ny + cnt - 1, :] = nx * ny + cnt, x, y, zz[i, j], mobs[j]
            VV_inv[0, i, j] = cnt
            VV_inv[1, i, j] = nx * ny + cnt
            cnt += 1

    # edges
    n_edges = ((nx - 1) * ny + (ny - 1) * nx) * 2 + nx * 2 + ny * 2 - 4
    EE = np.zeros((n_edges, 1 + 2), dtype=int)
    VV_EE = np.zeros((2, nx, ny, 6), dtype=int)

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
            EE[cnt - 1, :] = cnt, VV_inv[1, i, j], VV_inv[1, i, j + 1]
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

    # faces
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

    file += 'PARAMETER cnt = 0' + '\n'
    file += 'PARAMETER mobsum = 0' + '\n'
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
        # file += str(int(line[0])) + '\t' + str(line[1:])[1:-1] + '\n'
        file += str(int(line[0])) + '\t' + str(line[1:-1])[1:-1] + '\t vmob ' + str(line[-1]) + '\n'

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

    with open('notebooks/SE/SIM/datafile_ending.txt', 'r') as myfile:
        file += myfile.read()

    # % write to file
    with open('notebooks/SE/SIM/DEBER_datafile.fe', 'w') as myfile:
        myfile.write(file)


def create_datafile_non_period(yy_pre_pre, zz_pre_pre, mobs_pre):  # xx and zz in um !!!

    # plt.figure(dpi=300)
    # plt.plot(yy_pre_pre, zz_pre_pre)
    # plt.show()

    yy_pre = np.linspace(-mm.lx / 40, mm.lx / 40, 20)
    yy_beg = np.linspace(-mm.lx / 2, -mm.lx / 40 - 1, 40)
    yy_end = np.linspace(mm.lx / 40 + 1, mm.lx / 2, 40)
    yy_pre = np.concatenate([yy_beg, yy_pre, yy_end]) * 1e-3

    zz_pre = mcf.lin_lin_interp(yy_pre_pre, zz_pre_pre * 1e-3)(yy_pre / 1e-3)
    # mobs = mcf.lin_log_interp(yy_pre_pre, mobs_pre_pre)(yy_pre)
    z_max = np.max(zz_pre)

    def exp_gauss(x, a, b, c):
        return np.exp(a + b / c * np.exp(-x ** 2 / c ** 2))

    # popt = curve_fit(exp_gauss, yy_pre, mobs)[0]
    popt = curve_fit(exp_gauss, yy_pre_pre, mobs_pre, p0=[-30, 378, 150])[0]
    a, b, c = popt

    # plt.figure(dpi=300)
    # plt.plot(yy_pre_pre, mobs_pre, 'o-')
    # plt.plot(yy_pre_pre, exp_gauss(yy_pre_pre, *popt))
    # plt.plot(yy_pre, exp_gauss(yy_pre, -30.4, 658, 200))
    # plt.show()

    ny = len(yy_pre)
    nx = 2

    xx = np.linspace(-mm.ly / 2, mm.ly / 2, nx) * 1e-3
    yy = yy_pre
    zz = np.zeros((nx, ny))

    for i in range(len(xx)):
        zz[i, :] = zz_pre

    # plt.figure(dpi=300)
    # plt.plot(yy, zz_pre)
    # plt.show()

    volume = 0

    for i in range(len(xx) - 1):
        volume += np.trapz(zz[i, :], x=yy) * (xx[1] - xx[0])

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
            EE[cnt - 1, :] = cnt, VV_inv[1, i, j], VV_inv[1, i, j + 1]
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
            FF[cnt - 1, :] = cnt, VV_EE[0, i, j, 1], VV_EE[0, i, j + 1, 0], \
                             -VV_EE[0, i + 1, j, 1], -VV_EE[0, i, j, 0]
            cnt += 1

    for i in range(nx - 1):  # top
        for j in range(ny - 1):
            FF[cnt - 1, :] = cnt, VV_EE[1, i, j, 0], VV_EE[1, i + 1, j, 1], \
                             -VV_EE[1, i, j + 1, 0], -VV_EE[1, i, j, 1]
            cnt += 1

    for i in range(nx - 1):  # front
        FF[cnt - 1, :] = cnt, VV_EE[0, i, 0, 0], VV_EE[0, i + 1, 0, 4], \
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

    # file += 'PARAMETER lx = ' + str(mm.ly * 1e-3) + '\n'
    # file += 'PARAMETER ly = ' + str(mm.lx * 1e-3) + '\n'
    file += 'PARAMETER lx = 0.02' + '\n'
    file += 'PARAMETER ly = 3.3' + '\n'
    file += 'PARAMETER z_max = ' + str(z_max) + '\n'
    file += 'PARAMETER angle_surface = 55' + '\n'
    file += 'PARAMETER angle_mirror = 90' + '\n'

    file += 'PARAMETER par_a = ' + str(a) + '\n'
    file += 'PARAMETER par_b = ' + str(b) + '\n'
    file += 'PARAMETER par_c = ' + str(c) + '\n'

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

    file += '\n' + 'defmob_y := {foreach vertex vv do set vv vmob ' \
                   'exp(par_a + par_b / par_c * exp(-(vv.y*1e+3)^2 / par_c^2))}' + '\n\n'

    # % write to file
    with open('notebooks/SE/SE_input_3D_DEBER.fe', 'w') as myfile:
        myfile.write(file)


def run_evolver():
    os.system('evolver -f/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SIM/commands.txt ' +
              '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SIM/DEBER_datafile.fe')


def get_evolver_times_profiles():
    SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SIM/vlist_SIM.txt')

    times = []
    profiles = []
    beg = -1

    for i, line in enumerate(SE[1:]):
        if line[1] == line[2] == -100:
            print('read')
            now_time = line[0]
            times.append(now_time)
            profile = SE[beg + 1:i, 1:]
            
            profile = profile[np.where(np.abs(profile[:, 0]) < mapping.x_max * 1e-3)]
            profile = profile[np.where(profile[:, 1] > 0.03)]

            sort_inds = np.argsort(profile[:, 0])
            profile[:, 0] = profile[sort_inds, 0]
            profile[:, 1] = profile[sort_inds, 1]

            profiles.append(profile)
            beg = i

    pr_beg = profiles[0]
    pr_end = profiles[1]

    shift = ((pr_end[0, 1] - pr_beg[0, 1]) + (pr_end[-1, 1] - pr_beg[-1, 1])) / 2
    profiles[1][:, 1] -= shift

    return times, profiles


# %% read datafile
# yy_test = mapping.x_centers_5nm * 1e-3
# zz_test = np.ones(len(yy_test)) * mapping.l_z * 1e-3 *\
#           (1 - np.cos(2 * np.pi * mapping.x_centers_5nm / 3000) / 5)

# plt.figure(dpi=300)
# plt.plot(yy_test, zz_test, '.')
# plt.show()

# mobilities = np.ones(len(yy_test)) * 0.04
# create_datafile(yy_test, zz_test, mobilities)
# run_evolver()
