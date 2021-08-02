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
yy_um_0 = np.linspace(-15, -5 - 1e-4, 101)
yy_um_1 = np.linspace(-5, 5, 101)
yy_um_2 = np.linspace(5 + 1e-4, 15, 101)

zz_um_0 = np.ones(101) * 0.450
zz_um_1 = np.ones(101) * 0.350
zz_um_2 = np.ones(101) * 0.450

yy_um = np.concatenate((yy_um_0, yy_um_1, yy_um_2))
zz_um = np.concatenate((zz_um_0, zz_um_1, zz_um_2))

plt.figure(dpi=300)
plt.plot(yy_um, zz_um)
plt.show()

# %% make SE files
width_um = 1
path = 'notebooks/SE/datafile_blue_rect.fe'

mobilities = np.ones(len(yy_um)) * 2e-7

# %%
sf.create_datafile_no_mob_fit(yy_um, zz_um, width_um, mobilities, path)

# %%
SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist.txt')

SE = SE[np.where(
        np.logical_or(
            SE[:, 0] == 0,
            SE[:, 1] == -100
        ))]

plt.figure(dpi=300)

inds = np.where(SE[:, 1] == -100)[0]

now_pos = 0

for ind in inds:
    now_data = SE[(now_pos + 1):ind, :]
    plt.plot(now_data[:, 1], now_data[:, 2], '.')
    now_pos = ind

plt.ylim(0.02, 0.06)
plt.show()



