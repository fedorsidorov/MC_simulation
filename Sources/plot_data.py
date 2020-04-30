import matplotlib.pyplot as plt
import numpy as np


# %%
def plot_DATA(DATA, d_PMMA=0):
    fig, ax = plt.subplots(dpi=300)

    for tn in range(int(np.max(DATA[:, 0]))):
        if len(np.where(DATA[:, 0] == tn)[0]) == 0:
            continue
        beg = np.where(DATA[:, 0] == tn)[0][0]
        end = np.where(DATA[:, 0] == tn)[0][-1] + 1
        ax.plot(DATA[beg:end, 4], DATA[beg:end, 6])

    if d_PMMA != 0:
        points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
        ax.plot(points, np.zeros(len(points)), 'k')
        ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
