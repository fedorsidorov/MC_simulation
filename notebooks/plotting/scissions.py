import importlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import grid

grid = importlib.reload(grid)

const = importlib.reload(const)

# %%
kJmol_2_eV = 0.0103

MMA_bonds = {}

MMA_bonds["Oval"] = 13.62, 8
MMA_bonds["C'-O'"] = 815 * kJmol_2_eV, 4
MMA_bonds["C'-O"] = 420 * kJmol_2_eV, 2
MMA_bonds["C3-H"] = 418 * kJmol_2_eV, 12
MMA_bonds["C2-H"] = 406 * kJmol_2_eV, 4
MMA_bonds["C-C'"] = 373 * kJmol_2_eV, 2  ## 383-10 !!!!
MMA_bonds["O-C3"] = 364 * kJmol_2_eV, 2
MMA_bonds["C-C3"] = 356 * kJmol_2_eV, 2
MMA_bonds["C-C2"] = 354 * kJmol_2_eV, 4

n_bonds = len(MMA_bonds)
bond_inds = list(range(n_bonds))

bond_names = list(MMA_bonds.keys())

BDE_array = np.array(list(MMA_bonds.values()))

bonds_BDE = BDE_array[:, 0]
bonds_occ = BDE_array[:, 1]

# bond_names = list(MMA_bonds.keys())
Eb_Nel = np.array(list(MMA_bonds.values()))


# %%
def get_stairway(b_map_sc, EE=grid.EE):
    Eb_Nel_sc_list = []

    for val in b_map_sc.keys():
        Eb_Nel_sc_list.append([MMA_bonds[val][0], b_map_sc[val]])

    Eb_Nel_sc = np.array(Eb_Nel_sc_list)

    probs = np.zeros(len(EE))

    nums = np.zeros(len(EE))
    dens = np.zeros(len(EE))

    for i, e in enumerate(EE):

        num = 0

        for st in Eb_Nel_sc:
            if e >= st[0]:
                num += st[1]

        if num == 0:
            continue

        nums[i] = num

        den = 0

        for st in Eb_Nel:
            if e >= st[0]:
                den += st[1]

        dens[i] = den

        probs[i] = num / den

    return probs


# %%
_, _ = plt.subplots(dpi=600)
fig = plt.gcf()
fig.set_size_inches(3, 3)

font_size = 8

EE_low = np.linspace(3, 15, 1000)

plt.plot(EE_low, get_stairway({'C-C2': 4}, EE_low), label='30$^\circ$')
plt.plot(EE_low, get_stairway({'C-C2': 4, "C3-H": 4}, EE_low), '--', label='160$^\circ$')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.xlabel('$E$, эВ')
plt.ylabel('вероятность разрыва')

plt.xlim(3, 10)

plt.legend()

plt.grid()
# plt.show()

# %%
plt.savefig('stairway.tiff', bbox_inches='tight', dpi=600)

# %%
# font_size = 14
#
# plt.figure(figsize=[5, 5])
#
# matplotlib.rcParams['font.family'] = 'Times New Roman'
#
# plt.legend(fontsize=font_size, loc='lower right')
#
# ax = plt.gca()
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(font_size)
#
# for i in range(1, len(MMA_bonds)):
#     plt.loglog(grid.EE, scission_probs[:, i], label=bond_names[i])
#
# plt.xlabel('E, eV', fontsize=font_size)
# plt.ylabel('bond weight', fontsize=font_size)
#
# plt.xlim(1e+0, 1e+4)
# plt.ylim(1e-2, 1)
#
# plt.legend(loc='upper right', fontsize=font_size)
# plt.grid()
# plt.show()
