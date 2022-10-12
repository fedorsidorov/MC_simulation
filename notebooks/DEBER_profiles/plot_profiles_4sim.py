import numpy as np
import matplotlib.pyplot as plt

# %%
xx_1 = np.load('DEBER_profiles_4sim/150C_100s/xx_150C_100s.npy')
zz_1 = np.load('DEBER_profiles_4sim/150C_100s/zz_150C_100s.npy')

# plt.figure(dpi=600, figsize=[6.4, 4.8])
plt.figure(dpi=600, figsize=[6.4 / 1.9, 4.8 / 1.9])

# plt.plot(D_sim[:, 0], D_sim[:, 1], '.--', label='статья Дапора')
plt.plot(xx_1, zz_1 + 75)
plt.plot(xx_1, zz_1 + 100)

# plt.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], '.--', label='моделирование')

# plt.plot(D_exp[:, 0], D_exp[:, 1], 'o', label='эксперимент')
# plt.plot(energies_delta_nf_0p02[0], energies_delta_nf_0p02[1], '*', label='моделирование')

# plt.xlabel(r'$E$, эВ')
# plt.ylabel(r'$\delta$')
# plt.text(-300, 2.16, 'a)', fontsize=12)

# plt.xlim(0, 1500)
# plt.ylim(0, 2.5)

# plt.legend()
plt.grid()

plt.show()

