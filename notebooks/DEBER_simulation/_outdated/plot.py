import importlib
import numpy as np
import matplotlib.pyplot as plt
from mapping import mapping_3um_500nm as mm

# %%
d_PMMA = 500

profile = np.loadtxt('notebooks/DEBER_simulation/profile.txt')
xx, yy = profile[:, 0], profile[:, 1]

xx_366 = np.load('notebooks/DEBER_simulation/exp_profile_366/xx.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profile_366/zz.npy')

plt.figure(figsize=(5, 4), dpi=300)

plt.plot(mm.x_centers_50nm, np.ones(len(mm.x_centers_50nm)) * d_PMMA, '--', label='начальный уровень ПММА')
plt.plot(xx_366, zz_366, label='эксперимент')
plt.plot(xx, yy, label='моделирование')

plt.xlim(-1500, 1500)
plt.ylim(0, 600)

plt.xlabel('x, нм')
plt.ylabel('z, нм')
plt.legend(loc=1)
plt.grid()
# plt.show()
plt.savefig('profile.tiff', dpi=300, )
