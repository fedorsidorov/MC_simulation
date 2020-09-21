import matplotlib.pyplot as plt
import numpy as np
import importlib
from functions import reflow_functions as rf

rf = importlib.reload(rf)

# %%
sci_500 = np.load('notebooks/viscosity/final/sci_avg_arr_500.npy')
sci_700 = np.load('notebooks/viscosity/final/sci_avg_arr_700.npy')[10:]
sci_1000 = np.load('notebooks/viscosity/final/sci_avg_arr_1000.npy')[20:]
sci_1500 = np.load('notebooks/viscosity/final/sci_avg_arr_1500.npy')[30:]

Mw_500 = np.load('notebooks/viscosity/final/Mw_arr_500.npy')
Mw_700 = np.load('notebooks/viscosity/final/Mw_arr_700.npy')[10:]
Mw_1000 = np.load('notebooks/viscosity/final/Mw_arr_1000.npy')[20:]
Mw_1500 = np.load('notebooks/viscosity/final/Mw_arr_1500.npy')[30:]

# %%
fig = plt.figure(dpi=600)
fig.set_size_inches(4, 4)

plt.plot(sci_500 * 8, np.log10(Mw_500), 'o-', label='zip length = 500')
plt.plot(sci_700 * 8, np.log10(Mw_700), 'o-', label='zip length = 700')
plt.plot(sci_1000 * 8, np.log10(Mw_1000), 'o-', label='zip length = 1000')
plt.plot(sci_1500 * 8, np.log10(Mw_1500), 'o-', label='zip length = 1500')

plt.xlabel('число разрывов в объеме (10 нм)$^3$')
plt.ylabel('lg(Mw) (г/моль)')
plt.xlim(0, 2.5 * 8)
plt.ylim(4, 6)

plt.legend()
plt.grid()
plt.show()
# plt.savefig('Mw_for_zip_lens.png', bbox_inches='tight')





