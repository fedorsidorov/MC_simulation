import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import reflow_functions as rf
import matplotlib

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
MN = np.array([1e+3, 1e+4, 1e+5, 1e+6])
TT = np.linspace(100, 200, 21)

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogy(TT, rf.get_viscosity_experiment_Mn(TT, MN[0], 3.4, 1.4, Mn_edge=42000), '.-', label=r'$M_n = 1000$ g/mol')
    ax.semilogy(TT, rf.get_viscosity_experiment_Mn(TT, MN[1], 3.4, 1.4, Mn_edge=42000), '.-', label=r'$M_n = 10000$ g/mol')
    ax.semilogy(TT, rf.get_viscosity_experiment_Mn(TT, MN[2], 3.4, 1.4, Mn_edge=42000), '.-', label=r'$M_n = 100000$ g/mol')

    ax.set(xlabel=r'$T$, °C')
    ax.set(ylabel=r'$\eta$, Pa$\cdot$s')

    ax.legend(fontsize=7, loc='upper right')

    plt.xlim(100, 200)
    plt.ylim(1e+1, 1e+8)

    plt.savefig('eta_vary_T_Mn.jpg', dpi=600)
    plt.show()


# %%
MN = np.array([1e+3, 1e+4, 1e+5, 1e+6])
TT = np.linspace(100, 200, 21)

# with plt.style.context(['science', 'grid', 'russian-font']):
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)

ax.semilogy(TT, rf.get_viscosity_experiment_Mn(TT, MN[0], 3.4, 1.4, Mn_edge=42000), label=r'$M_\mathrm{n}$ = 1000 g/mol')
ax.semilogy(TT, rf.get_viscosity_experiment_Mn(TT, MN[1], 3.4, 1.4, Mn_edge=42000), label=r'$M_\mathrm{n}$ = 10000 g/mol')
ax.semilogy(TT, rf.get_viscosity_experiment_Mn(TT, MN[2], 3.4, 1.4, Mn_edge=42000), label=r'$M_\mathrm{n}$ = 100000 g/mol')

ax.set(xlabel=r'$T$, °C')
ax.set(ylabel=r'$\eta$, Pa$\cdot$s')

ax.legend(fontsize=9, loc='upper right')

plt.xlim(100, 200)
plt.ylim(1e+1, 1e+8)
plt.grid()

plt.savefig('eta_vary_T_Mn.jpg', dpi=600, bbox_inches='tight')
plt.show()
