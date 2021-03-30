import importlib
import numpy as np
from tqdm import tqdm
from functions import scission_functions as sf
import matplotlib.pyplot as plt
sf = importlib.reload(sf)

# %%
source = '/Volumes/Transcend/new_e_DATA_80nm/'
d_PMMA = 80

n_files = 1
primary_electrons_in_file = 100
file_cnt = 0

n_val_events = 0
E_dep = 0

weight = 0.35
progress_bar = tqdm(total=n_files, position=0)

for i in range(n_files):

    now_e_DATA = np.load(source + 'e_DATA_' + str(i) + '.npy')
    # now_e_DATA = now_e_DATA[np.where(now_e_DATA[:, 6] <= d_PMMA)]

    inds_val = np.where(now_e_DATA[:, 3] == 1)[0]

    now_e_DATA_val = now_e_DATA[inds_val, :]

    n_val_events += len(np.where(now_e_DATA[:, 3] == 1)[0])
    E_dep += np.sum(now_e_DATA[:, 7])

    progress_bar.update()

# %%
xx = now_e_DATA_val[:, 4]
zz = now_e_DATA_val[:, 6]

# %%
plt.figure(dpi=300)
plt.plot(now_e_DATA_val[:, 4], now_e_DATA_val[:, 6], '.')
plt.show()


# %%
def plot_e_DATA(e_DATA_arr, d_PMMA, E_cut=5):

    e_DATA_arr = e_DATA_arr[np.where(e_DATA_arr[:, -1] > E_cut)]
    fig, ax = plt.subplots(dpi=300)

    for e_id in range(int(np.max(e_DATA_arr[:, 0]) + 1)):
        inds = np.where(e_DATA_arr[:, 0] == e_id)[0]
        if len(inds) == 0:
            continue
        ax.plot(e_DATA_arr[inds, 4], e_DATA_arr[inds, 6], '-', linewidth='1')

    plt.plot(np.linspace(-500, 500, 100), np.ones(100) * d_PMMA, 'k-')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.grid()
    plt.show()


# %%
plot_e_DATA(now_e_DATA, d_PMMA)

