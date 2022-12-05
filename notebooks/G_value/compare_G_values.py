import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import constants as const
from mapping import mapping_harris as mapping
from functions import G_functions as Gf
from functions import MC_functions as mcf

mapping = importlib.reload(mapping)
const = importlib.reload(const)
mcf = importlib.reload(mcf)
Gf = importlib.reload(Gf)

# %%
weights = np.load('notebooks/G_value/samples/weights.npy')

TT_sim_1 = np.load('notebooks/G_value/samples/TT_sim_1.npy')
TT_sim_2 = np.load('notebooks/G_value/samples/TT_sim_2.npy')
TT_sim_3 = np.load('notebooks/G_value/samples/TT_sim_3.npy')
TT_sim_4 = np.load('notebooks/G_value/samples/TT_sim_4.npy')
TT_sim_5 = np.load('notebooks/G_value/samples/TT_sim_5.npy')

TT_theor_1 = np.load('notebooks/G_value/samples/TT_theor_1.npy')

plt.figure(dpi=300)
plt.plot(TT_sim_1, weights, '>', label='1')
plt.plot(TT_sim_2, weights, 'v', label='2')
plt.plot(TT_sim_3, weights, '<', label='3')
plt.plot(TT_sim_3, weights, '^', label='4')
plt.plot(TT_sim_3, weights, 'D', label='5')

plt.legend()
plt.grid()
plt.show()

# %%
tt = np.linspace(0, 200, 201)

ww_1 = mcf.lin_lin_interp(TT_sim_1, weights)(tt)
ww_2 = mcf.lin_lin_interp(TT_sim_2, weights)(tt)
ww_3 = mcf.lin_lin_interp(TT_sim_3, weights)(tt)
ww_4 = mcf.lin_lin_interp(TT_sim_4, weights)(tt)
ww_5 = mcf.lin_lin_interp(TT_sim_5, weights)(tt)

ww_theor_1 = mcf.lin_lin_interp(TT_theor_1, weights)(tt)

plt.figure(dpi=600, figsize=[4, 3])

plt.plot(tt, ww_1, label=r'моделирование $G_s$')
plt.plot(tt, ww_theor_1, '--', label=r'теоретическое $G_s$')
# plt.plot(tt, ww_2)
# plt.plot(tt, ww_3)
# plt.plot(tt, ww_4)
# plt.plot(tt, ww_5)

plt.xlabel(r'$T$, $^\circ$C')
plt.ylabel(r'$p_s$')

plt.xlim(0, 200)
plt.ylim(0.02, 0.12)

plt.legend()
plt.grid()

# plt.savefig('G_s.jpg', dpi=600, bbox_inches='tight')
plt.show()



