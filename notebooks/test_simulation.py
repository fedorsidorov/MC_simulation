# %%
import numpy as np
import matplotlib.pyplot as plt


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

    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    # plt.xlim(-50, 50)
    # plt.ylim(0, 50)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
now_DATA = np.load('data/e_DATA/DATA_test.npy')
# now_DATA_P = now_DATA[np.where(now_DATA[:, 2] == 0)]
plot_DATA(now_DATA, 500, E_cut=0)

# %%
now_DATA_Pn = np.load('data/e_DATA/DATA_test.npy')
ans = np.max(now_DATA[:, 0])
bns = np.max(now_DATA_Pn[:, 0])

a_inds = np.where(now_DATA[:, 0] == ans)[0]
b_inds = np.where(now_DATA_Pn[:, 0] == bns)[0]

cns = now_DATA[a_inds[0], :]
dns = now_DATA_Pn[b_inds[0], :]

# %%
plot_DATA(now_DATA, 500, E_cut=0)
