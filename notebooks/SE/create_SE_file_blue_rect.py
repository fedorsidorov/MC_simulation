import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import fourier_functions as ff
from functions import MC_functions as mcf
from functions import SE_functions as sf

mf = importlib.reload(mcf)
ff = importlib.reload(ff)
sf = importlib.reload(sf)

# %% original profile in 2011' paper
L_um = 22
H_1_um = 0.440
H_2_um = 0.335

yy_1_um = np.linspace(-30, -L_um / 2 - 1e-4, 201)
yy_2_um = np.linspace(-L_um / 2, L_um / 2, 201)
yy_3_um = np.linspace(L_um / 2 + 1e-4, 30, 201)

zz_1_um = np.ones(201) * H_1_um
zz_2_um = np.ones(201) * H_2_um
zz_3_um = np.ones(201) * H_1_um

yy_um = np.concatenate((yy_1_um, yy_2_um, yy_3_um))
zz_um = np.concatenate((zz_1_um, zz_2_um, zz_3_um))

plt.figure(dpi=300)
plt.plot(yy_um, zz_um)
plt.show()

# %% make SE files
width_um = 1
path = 'notebooks/SE/datafile_blue_rect_vmob_1p7e-3.fe'

# mobilities = np.ones(len(yy_um)) * 1
mobilities = np.ones(len(yy_um)) * 1.7e-3

# %%
sf.create_datafile_no_mob_fit(yy_um, zz_um, width_um, mobilities, path)

# %%
SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist.txt')

SE = SE[np.where(
        np.logical_or(
            np.abs(SE[:, 0]) < 0.1,
            SE[:, 1] == -100
        ))]

inds = np.where(SE[:, 1] == -100)[0]

now_pos = 0

plt.figure(dpi=300)

for ind in inds:
    now_data = SE[(now_pos + 1):ind, :]
    plt.plot(now_data[:, 1], now_data[:, 2], '.')
    now_pos = ind

# plt.ylim(0.425, 0.475)
plt.show()



