import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
# from mapping import mapping_3um_500nm as mm
from functions import MC_functions as mcf
from scipy.optimize import curve_fit

# mm = importlib.reload(mm)
mcf = importlib.reload(mcf)


# %%
def create_datafile_latest_um(yy, zz, width, mobs, path):  # xx and zz in um !!!

    yy[0] = yy[0].astype(np.float16)
    yy[-1] = yy[-1].astype(np.float16)

    lx = width
    ly = yy.max() - yy.min()

    zz_1d = zz
    z_max = np.max(zz_1d)

    ny = len(yy)
    nx = 2

    xx = np.linspace(-width / 2, width / 2, nx)
    zz = np.zeros((nx, ny))

    for i in range(len(xx)):
        zz[i, :] = zz_1d

    volume = 0

    for i in range(len(xx) - 1):
        volume += np.trapz(zz[i, :], x=yy) * (xx[1] - xx[0])

    n_vertices = nx * ny * 2
    VV = np.zeros((n_vertices, 1 + 3 + 1 + 1))  # + mobility + is_surface
    VV_inv = np.zeros((2, nx, ny), dtype=int)
    cnt = 1

    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            VV[cnt - 1, :] = cnt, x, y, 0, 0, 0
            # VV[nx * ny + cnt - 1, :] = nx * ny + cnt, x, y, zz[i, j], mobs[j], int(j % 2)
            VV[nx * ny + cnt - 1, :] = nx * ny + cnt, x, y, zz[i, j], mobs[j], int((-1)**j)
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
    file += 'define vertex attribute vmob real\n'
    file += 'define vertex attribute is_surface real\n\n'

    file += 'MOBILITY_TENSOR\n' + \
            'vmob  0     0\n' + \
            '0     vmob  0\n' + \
            '0     0  vmob\n\n'

    file += 'PARAMETER now_time = 0\n'

    file += 'PARAMETER lx = ' + str(lx) + '\n'
    file += 'PARAMETER ly = ' + str(ly) + '\n'

    file += 'PARAMETER y_min = ' + str(yy.min()) + '\n'
    file += 'PARAMETER y_max = ' + str(yy.max()) + '\n'

    file += 'PARAMETER z_max = ' + str(z_max) + '\n'
    file += 'PARAMETER angle_surface = 55\n'
    file += 'PARAMETER angle_mirror = 90\n'

    file += 'PARAMETER TENS_r = 33.5e-2\n'
    file += 'PARAMETER TENS_s = -TENS_r*cos((angle_surface)*pi/180)\n'
    # file += 'PARAMETER TENS_m = -TENS_r*cos((angle_mirror)*pi/180)\n\n'
    file += 'PARAMETER TENS_m = 0\n\n'

    file += '/*--------------------CONSTRAINTS START--------------------*/\n'

    file += 'constraint 1\n'
    file += 'formula: z = 0\n\n'

    file += 'constraint 11\n'
    file += 'formula: x = -lx/2\n\n'

    file += 'constraint 22\n'
    file += 'formula: y = -ly/2\n\n'

    file += 'constraint 222\n'
    file += 'formula: y = y_min\n\n'

    file += 'constraint 33\n'
    file += 'formula: x = lx/2\n\n'

    file += 'constraint 44\n'
    file += 'formula: y = ly/2\n\n'

    file += 'constraint 444\n'
    file += 'formula: y = y_max\n\n'

    file += 'constraint 5 nonpositive\n'
    file += 'formula: z = z_max\n\n'

    file += '/*--------------------CONSTRAINTS END--------------------*/\n\n'

    # % vertices
    file += '/*--------------------VERTICES START--------------------*/\n'
    file += 'vertices\n'

    for line in VV:
        # file += str(int(line[0])) + '\t' + str(line[1:-1])[1:-1] + '\t vmob ' + str(line[-1]) + '\n'
        file += str(int(line[0])) + '\t' + str(line[1:-2])[1:-1] + '\t vmob ' + str(line[-2]) +\
                '\t is_surface ' + str(line[-1]) + '\n'

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

    with open('notebooks/SE/datafile_ending_2022.txt', 'r') as myfile:
        file += myfile.read()

    # % write to file
    # with open('notebooks/SE/SIM/DEBER_datafile_total.fe', 'w') as myfile:
    with open(path, 'w') as myfile:
        myfile.write(file)


# %%
def run_evolver(file_full_path, commands_full_path):
    os.system('evolver -f' + commands_full_path + ' ' + file_full_path)
    # os.system('evolver -f/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/commands.txt ' +
    #           '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SIM/DEBER_datafile.fe')


# def get_evolver_times_profiles():
#     SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/SIM/vlist_SIM.txt')
#
#     times = []
#     profiles = []
#     beg = -1
#
#     for i, line in enumerate(SE[1:]):
#         if line[1] == line[2] == -100:
#             print('read')
#             now_time = line[0]
#             times.append(now_time)
#             profile = SE[beg + 1:i, 1:]
#
#             profile = profile[np.where(np.abs(profile[:, 0]) < mm.x_max * 1e-3)]
#             profile = profile[np.where(profile[:, 1] > 0.03)]
#
#             sort_inds = np.argsort(profile[:, 0])
#             profile[:, 0] = profile[sort_inds, 0]
#             profile[:, 1] = profile[sort_inds, 1]
#
#             profiles.append(profile)
#             beg = i
#
#     pr_beg = profiles[0]
#     pr_end = profiles[1]
#
#     shift = ((pr_end[0, 1] - pr_beg[0, 1]) + (pr_end[-1, 1] - pr_beg[-1, 1])) / 2
#     profiles[1][:, 1] -= shift
#
#     return times, profiles


def get_evolver_profile(path, y_max):

    SE = np.loadtxt(path)

    raw_profile = SE[1:, :]

    # raw_profile = raw_profile[np.where(raw_profile[:, 0] > mm.y_max * 1e-3 / 2)]
    raw_profile = raw_profile[np.where(raw_profile[:, 0] > y_max * 1e-3 / 2)]
    raw_profile = raw_profile[np.where(raw_profile[:, 2] > 0)]
    # raw_profile = raw_profile[np.where(raw_profile[:, 2] > mm.d_PMMA / 1000)]

    sort_inds = np.argsort(raw_profile[:, 1])

    final_profile = np.zeros((len(raw_profile), 2))

    final_profile[:, 0] = raw_profile[sort_inds, 1]
    final_profile[:, 1] = raw_profile[sort_inds, 2]

    return final_profile


# %%
# profile_s = get_evolver_profile('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_surface.txt')
# profile_i = get_evolver_profile('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_inner.txt')
# profile_t = get_evolver_profile('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_total.txt')
#
# plt.figure(dpi=300)
# plt.plot(profile_s[:, 0], profile_s[:, 1], '.-', ms=2)
# plt.plot(profile_i[:, 0], profile_i[:, 1], '.-', ms=2)
# plt.plot(profile_t[::2, 0], profile_t[::2, 1], '.-', ms=2)
# # plt.plot(profile_t[:, 0], profile_t[:, 1], '.-', ms=2)
#
# plt.xlim(-1.5, 1.5)
#
# plt.show()

# % read datafile
# yy_test = mapping.x_centers_5nm * 1e-3
# zz_test = np.ones(len(yy_test)) * mapping.l_z * 1e-3 *\
#           (1 - np.cos(2 * np.pi * mapping.x_centers_5nm / 3000) / 5)

# plt.figure(dpi=300)
# plt.plot(yy_test, zz_test, '.')
# plt.show()

# mobilities = np.ones(len(yy_test)) * 0.04
# create_datafile(yy_test, zz_test, mobilities)
# run_evolver()
