import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import grid
from notebooks.simple_PMMA_MC import simple_arrays as arr

grid = importlib.reload(grid)
arr = importlib.reload(arr)

# %%
layer_ind = 1

l_1 = 1 / arr.structure_total_IMFP[layer_ind]
l_2 = 1 / arr.structure_total_IMFP[1 - layer_ind]

E_ind = 900
d = 1

W1 = 1 / l_1[E_ind]
W2 = 1 / l_2[E_ind]

u1 = np.random.random()
free_path = - 1 / W1 * np.log(1 - u1)

if u1 < (1 - np.exp(- W1 * d)):
    free_path_corr = 1 / W1 * (-np.log(1 - u1))
    print('here')
else:
    free_path_corr = d + 1 / W2 * (-np.log(1 - u1) - W1 * d)

print(free_path, free_path_corr)

# %%
E_ind = 900
a1 = 10

W1 = 1 / l_1[E_ind]
W2 = 1 / l_2[E_ind]

s = np.linspace(0, 50, 100)

f = np.zeros(len(s))
F = np.zeros(len(s))

f[np.where(s < a1)[0]] = np.exp(-W1 * s[np.where(s < a1)[0]])
f[np.where(s >= a1)[0]] = np.exp(-W1 * a1) * np.exp(-W2 * (s[np.where(s >= a1)[0]] - a1))
f_1 = np.exp(-W1 * s)

F[np.where(s < a1)[0]] = 1 - np.exp(-W1 * s[np.where(s < a1)[0]])
F[np.where(s >= a1)[0]] = 1 - np.exp(-(W1 - W2) * a1 - W2 * s[np.where(s >= a1)[0]])
F_1 = 1 - np.exp(-W1 * s)

plt.figure(dpi=300)

plt.plot(s, f, 'o', label='f')
plt.plot(s, f_1, '-', label='f_1')
plt.plot(s, F, 'o', label='F')
plt.plot(s, F_1, '-', label='F_1')

plt.xlim(0, 50)
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

