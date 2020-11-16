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

T_C = '120'

mobs_500 = np.load('notebooks/viscosity/final_1/mobs_' + T_C + '/mobs_500.npy')
mobs_700 = np.load('notebooks/viscosity/final_1/mobs_' + T_C + '/mobs_700.npy')
mobs_1000 = np.load('notebooks/viscosity/final_1/mobs_' + T_C + '/mobs_1000.npy')
mobs_1500 = np.load('notebooks/viscosity/final_1/mobs_' + T_C + '/mobs_1500.npy')

fig = plt.figure(dpi=600)
fig.set_size_inches(4, 4)

plt.semilogy(sci_500 * 8, mobs_500, 'o-', label='zip length = 500')
plt.semilogy(sci_700 * 8, mobs_700, 'o-', label='zip length = 700')
plt.semilogy(sci_1000 * 8, mobs_1000, 'o-', label='zip length = 1000')
plt.semilogy(sci_1500 * 8, mobs_1500, 'o-', label='zip length = 1500')

plt.title('T = ' + T_C + '°C')
plt.xlabel('число разрывов в объеме (10 нм)$^3$')
plt.ylabel('подвижность вершин поверхности')

plt.xlim(0, 2.5 * 8)
# plt.ylim(1e-8, 1e-1)
plt.ylim(1e-8, 1e-0)
# plt.ylim(1e-6, 1e+1)

plt.legend(loc='lower right')
plt.grid()
# plt.show()
plt.savefig('Mw_for_zip_lens_' + T_C + '.png', bbox_inches='tight')
