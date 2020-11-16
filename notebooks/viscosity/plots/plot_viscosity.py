import matplotlib.pyplot as plt
import numpy as np
import importlib
from functions import reflow_functions as rf

rf = importlib.reload(rf)

# %%
T_C_arr = [120, 140, 160]
Mw = np.logspace(4, 6, 10)

fig = plt.figure(dpi=600)
fig.set_size_inches(4, 4)

for T_C in T_C_arr:

    etas = rf.get_viscosity_W(T_C, Mw)
    plt.loglog(Mw, etas, 'o-', label=str(T_C) + '°C')

plt.xlabel('($M_w$, г/моль')
plt.ylabel('$\eta$, Па·с')
plt.xlim(1e+4, 1e+6)
# plt.ylim(4, 6)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('figure.png', bbox_inches='tight')
