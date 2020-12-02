import matplotlib.pyplot as plt
import numpy as np
import importlib
from functions import reflow_functions as rf

rf = importlib.reload(rf)

# %%
T_C_arr = np.linspace(120, 160, 10)
Mw_arr = [1e+4, 1e+5, 1e+6]

fig = plt.figure(dpi=600)
fig.set_size_inches(4, 4)

for Mw in Mw_arr:

    etas = np.zeros(len(T_C_arr))
    for i, T_C in enumerate(T_C_arr):
        etas[i] = rf.get_viscosity_W(T_C, Mw)

    plt.semilogy(T_C_arr, etas, 'o-', label=str(Mw))

# plt.xlabel('($M_w$, г/моль')
# plt.ylabel('вязкость, Па с')
# plt.xlim(0, 2.5 * 8)
# plt.ylim(4, 6)

plt.legend()
plt.grid()
plt.show()
# plt.savefig('figure.png', bbox_inches='tight')





