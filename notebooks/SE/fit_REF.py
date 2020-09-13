import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %%
def func_1(x, k):
    return x * k


times = np.load('notebooks/SE/REF/t-scale/times.npy')
ind = 5
tt = np.load('notebooks/SE/REF/t-scale/tt_' + str(ind) + '.npy')
inds = np.load('notebooks/SE/REF/t-scale/inds_' + str(ind) + '.npy')
scales = times[inds]

alpha = curve_fit(func_1, scales, tt)[0]
xx = np.linspace(scales[0], scales[-1], 100)
yy = func_1(xx, alpha)

plt.figure(dpi=300)
plt.plot(scales, tt, 'o', label='simulation')
plt.plot(xx, yy, label='alpha = ' + str(alpha[0]))

plt.xlabel('time, s')
plt.ylabel('scale')
plt.legend()
plt.grid()
plt.show()
# plt.savefig(str(ind) + '.png')

# %%
etas_SI = np.load('notebooks/SE/REF/t-scale/etas_SI.npy')
etas = etas_SI[:6]
alphas = 22, 109, 483, 2323, 10666, 49914

plt.figure(dpi=300)
plt.loglog(etas, alphas, 'o-', label='simulation')
plt.xlabel('viscosity, Pa s')
plt.ylabel('time / scale, s')
plt.legend()
plt.grid()
plt.show()
# plt.savefig('alphas.png')


# %%
def func_2(x, k, b):
    return np.exp(np.log(x) * k + b)


kk, bb = curve_fit(func_2, etas, alphas)[0]

xx = np.logspace(3, 7, 100)
yy = func_2(xx, kk, bb)

plt.figure(dpi=300)
plt.loglog(xx, yy, label='simulation')

plt.loglog(etas, alphas, 'o-', label='simulation')

plt.xlabel('viscosity, Pa s')
plt.ylabel('time / scale, s')
plt.legend()
plt.grid()
plt.show()
# plt.savefig(



