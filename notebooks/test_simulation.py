# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def plot_DATA(DATA, d_PMMA=0, E_cut=5):
    print('initial size =', len(DATA))
    DATA_cut = DATA[np.where(DATA[:, 9] > E_cut)]
    print('cut DATA size =', len(DATA_cut))
    fig, ax = plt.subplots(dpi=300)

    for tn in range(int(np.max(DATA_cut[:, 0]))):
        if len(np.where(DATA_cut[:, 0] == tn)[0]) == 0:
            continue
        now_DATA_cut = DATA_cut[np.where(DATA_cut[:, 0] == tn)]
        ax.plot(now_DATA_cut[:, 4], now_DATA_cut[:, 6])

    if d_PMMA != 0:
        points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
        ax.plot(points, np.zeros(len(points)), 'k')
        ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


now_DATA = np.load('data/e_DATA/DATA_0.npy')
plot_DATA(now_DATA, 0, E_cut=0)

