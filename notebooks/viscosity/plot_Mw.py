import matplotlib.pyplot as plt
import numpy as np

# %%
Mw_500 = np.load('notebooks/viscosity/Mw_arr_500.npy')
Mw_700 = np.load('notebooks/viscosity/Mw_arr_700.npy')
Mw_1000 = np.load('notebooks/viscosity/Mw_arr_1000.npy')
Mw_1500 = np.load('notebooks/viscosity/Mw_arr_1500.npy')

sci_500 = np.load('notebooks/viscosity/sci_avg_arr_500.npy')
sci_700 = np.load('notebooks/viscosity/sci_avg_arr_700.npy')
sci_1000 = np.load('notebooks/viscosity/sci_avg_arr_1000.npy')
sci_1500 = np.load('notebooks/viscosity/sci_avg_arr_1500.npy')

plt.figure(dpi=300)
# plt.plot([0], [677985], 'o')
plt.semilogy(sci_500, Mw_500)
plt.semilogy(sci_700, Mw_700)
plt.semilogy(sci_1000, Mw_1000)
plt.show()
