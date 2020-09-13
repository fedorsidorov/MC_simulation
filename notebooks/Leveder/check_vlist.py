import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import reflow_functions as rf
from functions import MC_functions as mf

mf = importlib.reload(mf)
rf = importlib.reload(rf)

# %%
SE = np.loadtxt('notebooks/SE/SIM/vlist_SIM.txt')[1:, :]

times = []
profiles = []
beg = -1

for i, line in enumerate(SE):
    if line[1] == line[2] == -100:
        now_time = line[0]
        times.append(now_time)
        profile = SE[beg+1:i, 1:]
        # profile = profile[profile[:, 1].argsort()]
        # profile = profile[profile[:, 0].argsort()]
        profile = profile[np.where(np.abs(profile[:, 0]) < 5)]
        profile = profile[np.where(profile[:, 1] > 0.02)]
        profile[:, 1] -= (np.max(profile[:, 1]) + np.min(profile[:, 1]))/2 - 0.05
        profiles.append(profile)
        beg = i

# %%
plt.figure(dpi=300)
plt.plot(profiles[0][:, 0], profiles[0][:, 1], '.')
plt.plot(profiles[1][:, 0], profiles[1][:, 1], '.')
plt.show()
