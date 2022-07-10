import importlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# %%
def linear_func(xx, a):
    return a*xx


SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/REF_Leveder/vlist_REF_scale_1e-3.txt')
SE = SE[np.where(np.logical_or(np.logical_and(SE[:, 0] == 0, SE[:, 2] > 0.02), SE[:, 1] == -100))]

inds = np.where(SE[:, 1] == -100)[0]
scales = []
profiles = []
now_pos = 0

for ind in inds[1:]:
    now_scale = SE[now_pos, 0]
    now_profile = SE[(now_pos + 1):ind, 1:]
    scales.append(now_scale)
    profiles.append(now_profile)
    now_pos = ind

scales = np.array(scales)


# %%
etas_SI = np.array((1e+2, 3.1e+2, 1e+3, 3.1e+3, 1e+4, 3.1e+4, 1e+5, 3.1e+5,
                    1e+6, 3.1e+6, 1e+7, 3.1e+7, 1e+8, 3.1e+8, 1e+9))

# now_eta = etas_SI[0]
# tt = [0.1, 0.5, 1, 2, 3]
# inds = [7, 14, 28, 54, 83]

# now_eta = etas_SI[1]
# tt = [1, 2, 4, 6, 10]
# inds = [10, 18, 36, 54, 90]

# now_eta = etas_SI[2]
# tt = [4, 8, 14, 20, 30]
# inds = [11, 22, 38, 55, 85]

# now_eta = etas_SI[3]
# tt = [10, 25, 40, 55, 90]
# inds = [9, 22, 36, 50, 82]

# now_eta = etas_SI[4]
# tt = [30, 80, 140, 230, 400]
# inds = [9, 22, 39, 65, 110]

# now_eta = etas_SI[5]
# tt = [80, 250, 500, 700, 1200]
# inds = [8, 22, 45, 63, 110]

# now_eta = etas_SI[6]
# tt = [200, 800, 1300, 2000, 3800]
# inds = [8, 22, 37, 56, 110]

# now_eta = etas_SI[7]
# tt = [600, 2000, 4000, 6000, 10000]
# inds = [8, 18, 36, 54, 90]

now_eta = etas_SI[8]
tt = [2000, 7000, 12000, 20000, 30000]
inds = [7, 19, 34, 55, 86]

now_scales = scales[inds]

xx = np.linspace(0, 40000, 1000)

popt, pcov = curve_fit(linear_func, tt, now_scales)
yy = linear_func(xx, *popt)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    # ax.plot(now_scales, tt, 'o', label=r'simulation')
    # ax.plot(xx, yy, 'r', label=r'time = $\alpha$$\cdot$scale fit')

    ax.plot(tt, now_scales, '.', label=r'simulation')
    ax.plot(xx, yy, 'r', label=r'scale = $\alpha$$\cdot$time fit')

    ax.plot([-1], [-1], 'w', label=r'$\alpha$=' + str(format(popt[0], '.2e')))

    ax.set(title=r'$\eta$ = ' + str(format(now_eta, '.1e')) + ' Pa$\cdot$s')
    ax.set(xlabel=r'time, s')
    ax.set(ylabel=r'scale')
    ax.legend(fontsize=7, loc='lower right')

    # plt.xlim(0, 1)
    # plt.xlim(0, 1.25)
    # plt.xlim(0, 1.5)

    # plt.ylim(0, 4)
    # plt.ylim(0, 12)
    # plt.ylim(0, 40)
    # plt.ylim(0, 120)
    # plt.ylim(0, 500)
    # plt.ylim(0, 1500)
    # plt.ylim(0, 5000)
    # plt.ylim(0, 12000)
    # plt.ylim(0, 40000)

    # plt.xlim(0, 4)
    # plt.xlim(0, 12)
    # plt.xlim(0, 40)
    # plt.xlim(0, 120)
    # plt.xlim(0, 500)
    # plt.xlim(0, 1500)
    # plt.xlim(0, 5000)
    # plt.xlim(0, 12000)
    plt.xlim(0, 40000)

    plt.ylim(0, 1)
    # plt.ylim(0, 1.2)

    plt.savefig('figures_final/alpha_' + str(int(now_eta)) + '.jpg', dpi=600)
    # plt.show()
