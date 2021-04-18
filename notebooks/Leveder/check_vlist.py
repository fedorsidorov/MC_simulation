import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import reflow_functions as rf
from functions import MC_functions as mf

mf = importlib.reload(mf)
rf = importlib.reload(rf)

# %%
SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist.txt')

SE = SE[np.where(
        np.logical_or(
            SE[:, 0] == 0,
            SE[:, 1] == -100
        ))]

plt.figure(dpi=300)

inds = np.where(SE[:, 1] == -100)[0]

now_pos = 0

for ind in inds:
    now_data = SE[(now_pos + 1):ind, :]
    plt.plot(now_data[:, 1], now_data[:, 2], '.')
    now_pos = ind

plt.ylim(0.02, 0.06)
plt.show()
