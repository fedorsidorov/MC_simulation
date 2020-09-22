import matplotlib.pyplot as plt
import numpy as np
import importlib
from functions import reflow_functions as rf

rf = importlib.reload(rf)

# %%
sci_500 = np.load('notebooks/viscosity/final_1/sci_500.npy')
sci_700 = np.load('notebooks/viscosity/final_1/sci_700.npy')
sci_1000 = np.load('notebooks/viscosity/final_1/sci_1000.npy')
sci_1500 = np.load('notebooks/viscosity/final_1/sci_1500.npy')

Mw_500 = np.load('notebooks/viscosity/final_1/Mw_500.npy')
Mw_700 = np.load('notebooks/viscosity/final_1/Mw_700.npy')
Mw_1000 = np.load('notebooks/viscosity/final_1/Mw_1000.npy')
Mw_1500 = np.load('notebooks/viscosity/final_1/Mw_1500.npy')

T_C = 120
# T_C = 140
# T_C = 160

mobs_500 = rf.move_Mw_to_mob(T_C, Mw_500)
mobs_700 = rf.move_Mw_to_mob(T_C, Mw_700)
mobs_1000 = rf.move_Mw_to_mob(T_C, Mw_1000)
mobs_1500 = rf.move_Mw_to_mob(T_C, Mw_1500)

fig = plt.figure(dpi=600)
fig.set_size_inches(4, 4)

plt.semilogy(sci_500 * 8, mobs_500, 'o-', label='zip length = 500')
plt.semilogy(sci_700 * 8, mobs_700, 'o-', label='zip length = 700')
plt.semilogy(sci_1000 * 8, mobs_1000, 'o-', label='zip length = 1000')
plt.semilogy(sci_1500 * 8, mobs_1500, 'o-', label='zip length = 1500')

plt.xlabel('число разрывов в объеме (10 нм)$^3$')
plt.ylabel('подвижность вершин поверхности')
plt.xlim(0, 2.5 * 8)
# plt.ylim(4, 6)

plt.legend()
plt.grid()
plt.show()
# plt.savefig('Mw_for_zip_lens_120.png', bbox_inches='tight')
