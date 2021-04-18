import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %%
def func_1(x, k):
    return x * k


etas_SI = np.load('notebooks/Leveder/times_scales/etas_SI.npy')
ind = 8

scales = np.load('notebooks/Leveder/times_scales/scales.npy')
now_tt = np.load('notebooks/Leveder/times_scales/tt_' + str(ind) + '.npy')
inds = np.load('notebooks/Leveder/times_scales/inds_' + str(ind) + '.npy')
now_scales = scales[inds]

alpha = curve_fit(func_1, now_scales, now_tt)[0]
xx = np.linspace(now_scales[0], now_scales[-1], 100)
yy = func_1(xx, alpha)

plt.figure(dpi=300)
plt.plot(now_scales, now_tt, 'o', label='simulation')
plt.plot(xx, yy, label='alpha = ' + str(int(alpha[0] * 100) / 100))

plt.xlabel('scale')
plt.ylabel('time, s')
plt.title(r'$\eta$ = ' + str(etas_SI[ind]))

plt.legend()
plt.grid()
plt.show()
# plt.savefig(str(ind) + '.png')

# %%
etas = etas_SI[:9]
alphas = 3.62, 11.1, 35.79, 110.14, 360.87, 1098.27, 3482.18, 11092.91, 35329.94

plt.figure(dpi=300)
plt.loglog(etas, alphas, 'o-', label='simulation')
plt.xlabel(r'viscosity, Pa s')
plt.ylabel(r'time / scale')
plt.legend()
plt.grid()
plt.show()
# plt.savefig('alphas.png')


# %%
def func_2(x, C, k):
    return C * x**k


CC, kk = curve_fit(func_2, etas, alphas)[0]

xx = np.logspace(2, 6, 100)
# yy = func_2(xx, CC, kk)
yy = func_2(xx, 0.0381, 0.9946)

plt.figure(dpi=300)
plt.loglog(etas, alphas, 'o-', label='simulation')
plt.loglog(xx, yy, label=r'time / scale = C * $\eta ^ k$, C = 0.0381, k = 0.9946')

plt.xlabel('viscosity, Pa s')
plt.ylabel(r'time / scale')
plt.legend()
plt.grid()
plt.show()
# plt.savefig('alphas_fit.png')
