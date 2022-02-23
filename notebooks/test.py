import numpy as np
import matplotlib.pyplot as plt

# %%
dT = 0.1

TT = np.arange(0, 10, dT)
FF = np.sin(TT * 2 * np.pi / 1)

FF_1d = np.zeros(len(TT))
FF_1d[0] = FF[0]

for i in range(1, len(TT)):
    first_diff = 2 * np.pi * np.cos(TT[i-1] * 2 * np.pi / 1) * dT +\
        (2 * np.pi)**2 * (-np.sin(TT[i-1] * 2 * np.pi / 1)) * dT**2 / 2 + \
        (2 * np.pi) ** 3 * (-np.cos(TT[i - 1] * 2 * np.pi / 1)) * dT ** 3 / 6 + \
        (2 * np.pi) ** 4 * (np.sin(TT[i - 1] * 2 * np.pi / 1)) * dT ** 5 / 24
    FF_1d[i] = FF_1d[i-1] + first_diff

plt.figure(dpi=300)
plt.plot(TT, FF)
plt.plot(TT, FF_1d, '--')
plt.grid()
plt.show()




