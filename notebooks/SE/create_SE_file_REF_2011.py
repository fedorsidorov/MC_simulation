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
l0_um = 2

yy_um_0_paper = np.array((0, 0.464, 0.513, 1.5, 1.55, 2))
zz_nm_0_paper = np.array((27.5, 27.5, 55.2, 55.2, 27.5, 27.5))

yy_um_0 = np.concatenate((yy_um_0_paper, yy_um_0_paper + l0_um, yy_um_0_paper + 2*l0_um, yy_um_0_paper + 3*l0_um,
                          yy_um_0_paper + 4*l0_um))
yy_um_0 -= (yy_um_0.max() - yy_um_0.min()) / 2

zz_nm_0 = np.concatenate((zz_nm_0_paper, zz_nm_0_paper, zz_nm_0_paper, zz_nm_0_paper, zz_nm_0_paper))

yy_nm = yy_um_0 * 1e+3
zz_nm = zz_nm_0

plt.figure(dpi=300)
plt.plot(yy_nm, zz_nm)
plt.show()

# %% make SE files
width = 200
path = 'notebooks/SE/datafile_REF.fe'

yy_SE = np.linspace(yy_nm.min(), yy_nm.max(), 200)
zz_SE = mcf.lin_lin_interp(yy_nm, zz_nm)(yy_SE)

width *= 1e-3
yy_SE *= 1e-3
zz_SE *= 1e-3

mobilities = np.ones(len(yy_SE))

# plt.figure(dpi=300)
# plt.plot(yy_SE, zz_SE, '.')
# plt.show()

sf.create_datafile_no_mob_fit(yy_SE, zz_SE, width, mobilities, path)

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



