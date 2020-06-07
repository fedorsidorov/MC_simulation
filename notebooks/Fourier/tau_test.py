import numpy as np
import matplotlib.pyplot as plt


# %%
# l_period = 10e-6
# l_period = 500e-9
A = 1e-19  # J
gamma = 34e-3  # N / m
# h0 = 307e-9  # m
# h0 = 4e-9  # m
# eta = 5.51e+6  # Pa * s
eta = 1.3e+7  # Pa * s


def get_tau(l0, h0, num=1):
    if num == 0:
        return np.inf
    part_1 = (num * 2 * np.pi / l0)**2 * A / (6 * np.pi * h0 * eta)
    part_2 = (num * 2 * np.pi / l0)**4 * gamma * h0**3 / (3 * eta)
    return 1 / (part_1 + part_2)


h0_array = np.logspace(0, 2, 100) * 1e-9

plt.figure(dpi=300)

for l in np.array((50, 70, 125, 250, 500)) * 1e-9:
    plt.loglog(h0_array * 1e+9, get_tau(l, h0_array))

# plt.show()


tau_50nm = np.loadtxt('data/reflow/50nm.txt')
tau_70nm = np.loadtxt('data/reflow/70nm.txt')
tau_125nm = np.loadtxt('data/reflow/125nm.txt')
tau_250nm = np.loadtxt('data/reflow/250nm.txt')
tau_500nm = np.loadtxt('data/reflow/500nm.txt')

# plt.figure(dpi=300)
plt.loglog(tau_50nm[:, 0], tau_50nm[:, 1], 'o')
plt.loglog(tau_70nm[:, 0], tau_70nm[:, 1], 'o')
plt.loglog(tau_125nm[:, 0], tau_125nm[:, 1], 'o')
plt.loglog(tau_250nm[:, 0], tau_250nm[:, 1], 'o')
plt.loglog(tau_500nm[:, 0], tau_500nm[:, 1], 'o')
plt.show()


