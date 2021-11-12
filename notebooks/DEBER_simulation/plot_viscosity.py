import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_3p3um_80nm as mm
from functions import MC_functions as mcf
from functions import DEBER_functions as df
from functions import SE_functions as ef
from functions import reflow_functions as rf

const = importlib.reload(const)
mm = importlib.reload(mm)
mcf = importlib.reload(mcf)
df = importlib.reload(df)
ef = importlib.reload(ef)
rf = importlib.reload(rf)

xx_eta_2 = np.load('xx_eta_2.npy')
eta_2 = np.load('eta_2.npy')

# %%
xx = mm.x_centers_20nm
Mn_arr = np.load('Mn_array.npy')

eta_arr = rf.get_viscosity_experiment_Mn(160, Mn_arr, 3.4)

# plt.figure(dpi=300)

fontsize = 10

_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
fig.set_size_inches(4, 3)

plt.semilogy(xx * 1.5151, eta_arr, label='simulated: Mn, T')
plt.semilogy(xx_eta_2, eta_2, label='empirical: Mn, T, monomer')

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)

# plt.ylim(1e+4, 1e+8)
plt.ylim(1e+1, 1e+9)

plt.xlabel('x, nm', fontsize=fontsize)
plt.ylabel(r'$\eta$, Pa s', fontsize=fontsize)
plt.legend(fontsize=fontsize)

plt.xlim(-2500, 2500)

plt.grid()
# plt.show()
plt.savefig('eta_2.jpg', bbox_inches='tight')
