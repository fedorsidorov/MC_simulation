import importlib

import matplotlib.pyplot as plt
import numpy as np
from functions import G_functions as Gf
import constants as const

# %%
weights = np.load('data/choi_G_values/weights.npy')
G_exp_array = np.load('data/choi_G_values/G_exp_array.npy')

TT = list(range(20, 191, 15))
weights_TT = np.zeros(len(TT))

for i, T in enumerate(TT):

    G_required = Gf.get_G_value(T)
    ind = np.argmin(np.abs(G_exp_array - G_required))
    weights_TT[i] = weights[ind]

fig, ax = plt.subplots(dpi=300)
fig.set_size_inches(5, 4)

plt.plot(TT, weights_TT, '-o')

plt.xlabel('T, °C')
plt.ylabel('remote scission probability')

plt.yticks(np.array((0.2, 0.25, 0.3, 0.35, 0.4, 4)), ('0.2', '0.25', '0.3', '0.35', '0.4'))

# plt.legend()
plt.grid()
plt.xlim(0, 200)
plt.ylim(0.200, 0.425)

plt.show()

# plt.savefig('weights.tiff')

