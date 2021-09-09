import numpy as np
import importlib
from functions._outdated import SE_functions as sef
from mapping._outdated import mapping_viscosity_80nm as mm
import matplotlib.pyplot as plt

sef = importlib.reload(sef)
mm = importlib.reload(mm)

# %%
xx = mm.x_centers_25nm
l0 = mm.l_y
zz = np.ones(len(xx)) * mm.l_z * (1 - (1/4) * np.cos(2 * np.pi * xx / mm.l_x))

mobs = np.ones(len(xx))

plt.figure(dpi=300)
plt.plot(xx, np.zeros(len(zz)))
plt.plot(xx, zz)
plt.show()

# %%
sef.create_datafile_non_period(xx, zz, l0, mobs)





