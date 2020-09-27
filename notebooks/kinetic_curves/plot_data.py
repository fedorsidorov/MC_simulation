import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
data_125C = np.loadtxt('data/kinetic_curves/kin_curve_125C_80nm.txt')
data_150C = np.loadtxt('data/kinetic_curves/kin_curve_150C_80nm.txt')
data_170C = np.loadtxt('data/kinetic_curves/kin_curve_170C_80nm.txt')

data_160C = np.loadtxt('data/kinetic_curves/kin_curve_160C_900nm.txt')

# dose_list = np.load('notebooks/kinetic_curves/sim_data/dose_list.npy')
dose_list = np.load('notebooks/kinetic_curves/sim_data/dose_list_900nm.npy')

L_125C_300 = np.load('notebooks/kinetic_curves/sim_data/L_norm_125C_300.npy')
L_125C_600 = np.load('notebooks/kinetic_curves/sim_data/L_norm_125C_600.npy')
L_150C_500 = np.load('notebooks/kinetic_curves/sim_data/L_norm_150C_500.npy')
L_150C_1800 = np.load('notebooks/kinetic_curves/sim_data/L_norm_150C_1800.npy')
L_170C_1000 = np.load('notebooks/kinetic_curves/sim_data/L_norm_170C_1000.npy')
L_170C_4500 = np.load('notebooks/kinetic_curves/sim_data/L_norm_170C_4500.npy')

L_160C_5500 = np.load('notebooks/kinetic_curves/sim_data/L_norm_160C_5500_900nm.npy')

# dose_list_depol = np.load('notebooks/kinetic_curves/sim_data/dose_list_depol_80nm.npy')
# L_norm_depol = np.load('notebooks/kinetic_curves/sim_data/L_norm_depol_80nm.npy')

dose_list_depol = np.load('notebooks/kinetic_curves/sim_data/dose_list_depol_900nm.npy')
L_norm_depol = np.load('notebooks/kinetic_curves/sim_data/L_norm_depol_900nm.npy')

plt.figure(dpi=300)
# plt.plot(data_125C[:, 0], data_125C[:, 1], '*--', label='exp 125 °C')
# plt.plot(data_150C[:, 0], data_150C[:, 1], '*--', label='exp 150 °C')
plt.plot(data_160C[:, 0], data_160C[:, 1], '*--', label='exp 150 °C')
# plt.plot(data_170C[:, 0], data_170C[:, 1], '*--', label='exp 170 °C')

# plt.plot(dose_list, L_125C_300, label='sim zip length = 300')
# plt.plot(dose_list, L_125C_600)

# plt.plot(dose_list, L_150C_500, label='sim zip length = 500')
# plt.plot(dose_list, L_150C_1800)

plt.plot(dose_list, L_160C_5500, label='sim zip length = 5500')
# plt.plot(dose_list, L_160C_5500)

# plt.plot(dose_list, L_170C_1000, label='sim zip length = 1000')
# plt.plot(dose_list, L_170C_4500)

plt.plot(dose_list_depol, L_norm_depol, label='sim depolymerization')

# plt.title('80nm PMMA, 150 °C')
plt.title('900nm PMMA, 160 °C')
plt.xlabel('D, $\mu$C/cm$^2$')
plt.ylabel('L/L$_0$')

# plt.xlim(0, 20)
plt.xlim(0, 5)
plt.ylim(0, 1)
plt.grid()
plt.legend()

plt.show()
# plt.savefig('80nm_150C.png', dpi=300)
