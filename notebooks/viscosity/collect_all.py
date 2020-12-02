import matplotlib.pyplot as plt
import numpy as np
import importlib
from functions import reflow_functions as rf

rf = importlib.reload(rf)

# %%
was_500 = np.loadtxt('notebooks/viscosity/was/blue.txt')
was_700 = np.loadtxt('notebooks/viscosity/was/orange.txt')
was_1000 = np.loadtxt('notebooks/viscosity/was/green.txt')
was_1500 = np.loadtxt('notebooks/viscosity/was/red.txt')

# %%
plt.figure(dpi=300)

for arr in [was_500, was_700, was_1000, was_1500]:
    plt.semilogy(arr[:, 0], arr[:, 1])

plt.show()

# %%
Mw_500 = np.load('notebooks/viscosity/was/Mw_arr_500.npy')
Mw_700 = np.load('notebooks/viscosity/was/Mw_arr_700.npy')[6:12]
Mw_1000 = np.load('notebooks/viscosity/was/Mw_arr_1000.npy')[12:18]
Mw_1500 = np.load('notebooks/viscosity/was/Mw_arr_1500.npy')[18:]

sci_500 = np.load('notebooks/viscosity/was/sci_avg_arr_500.npy')
sci_700 = np.load('notebooks/viscosity/was/sci_avg_arr_700.npy')[6:12]
sci_1000 = np.load('notebooks/viscosity/was/sci_avg_arr_1000.npy')[12:18]
sci_1500 = np.load('notebooks/viscosity/was/sci_avg_arr_1500.npy')[18:]

plt.figure(dpi=300)
plt.semilogy(sci_500, Mw_500, 'o-')
plt.semilogy(sci_700, Mw_700, 'o-')
plt.semilogy(sci_1000, Mw_1000, 'o-')
plt.semilogy(sci_1500, Mw_1500, 'o-')
plt.show()

# %%
T_C = 120
mobs_500 = rf.move_Mw_to_mob(T_C, Mw_500)
mobs_700 = rf.move_Mw_to_mob(T_C, Mw_700)
mobs_1000 = rf.move_Mw_to_mob(T_C, Mw_1000)
mobs_1500 = rf.move_Mw_to_mob(T_C, Mw_1500)

mobs_500_f = np.concatenate((mobs_500[:-1], was_500[:, 1]))
mobs_700_f = np.concatenate((mobs_700[:-1], was_700[:, 1]))
mobs_1000_f = np.concatenate((mobs_1000[:-1], was_1000[:, 1]))
mobs_1500_f = np.concatenate((mobs_1500[:-1], was_1500[:, 1]))

sci_500_f = np.concatenate((sci_500[:-1], was_500[:, 0]))
sci_700_f = np.concatenate((sci_700[:-1], was_700[:, 0]))
sci_1000_f = np.concatenate((sci_1000[:-1], was_1000[:, 0]))
sci_1500_f = np.concatenate((sci_1500[:-1], was_1500[:, 0]))


# %%
plt.figure(dpi=300)
plt.semilogy(sci_500_f, mobs_500_f, 'o-')
plt.semilogy(sci_700_f, mobs_700_f, 'o-')
plt.semilogy(sci_1000_f, mobs_1000_f, 'o-')
plt.semilogy(sci_1500_f, mobs_1500_f, 'o-')
plt.show()



