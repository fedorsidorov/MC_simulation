import numpy as np
import matplotlib.pyplot as plt

# %%
plt.figure(dpi=300)

# p_1 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/initial/1.csv', delimiter=',', skiprows=5)
# plt.plot(p_1[:, 0] - 25000 - 1000 + 700, p_1[:, 1] + 14, label='initial')

p_1 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/initial/2.csv', delimiter=',', skiprows=5)
plt.plot(p_1[:, 0] - 25000 - 1000, p_1[:, 1], label='initial')

# p_2 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/34h_200C/1.csv', delimiter=',', skiprows=5)
# plt.plot(p_2[:, 0] - 12700 - 25000 - 1000, p_2[:, 1] - 16, label='34 hours at 200 C')

p_2 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/34h_200C/2.csv', delimiter=',', skiprows=5)
plt.plot(p_2[:, 0] - 12700 - 25000 - 1000 + 2600, p_2[:, 1] - 16, label='34 hours at 200 C')

# p_2 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/34h_200C/1.csv', delimiter=',', skiprows=5)
# plt.plot(p_2[:, 0] - 12700 - 25000 - 1000, p_2[:, 1] - 16, label='34 hours at 200 C')

# p_2 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/34h_200C/1.csv', delimiter=',', skiprows=5)
# plt.plot(p_2[:, 0] - 12700 - 25000 - 1000, p_2[:, 1] - 16, label='34 hours at 200 C')

# plt.grid()
# plt.legend()
# plt.show()


# SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist.txt')
SE = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/vlist_vmob_1.txt')

SE = SE[np.where(
        np.logical_or(
            np.abs(SE[:, 0]) < 0.1,
            SE[:, 1] == -100
        ))]

inds = np.where(SE[:, 1] == -100)[0]

now_pos = 0

now_i = 20

for i, ind in enumerate(inds):
    now_data = SE[(now_pos + 1):ind, :]
    ans = now_data

    if i == now_i:
        plt.plot(now_data[:, 1] * 1000, now_data[:, 2] * 1000, '.')

    # plt.plot(now_data[:, 1] * 1000, now_data[:, 2] * 1000, '.')
    # plt.plot(now_data[:, 1] * 1000 - 800, now_data[:, 2] * 1000, '.')
    # plt.plot(now_data[:, 1] * 1000 + 1100, now_data[:, 2] * 1000, '.')
    # plt.plot(now_data[:, 1] * 1000 * 1.08, now_data[:, 2] * 1000, '.')

    now_pos = ind

plt.grid()
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.legend()

plt.xlim(-20000, 20000)
plt.ylim(300, 500)

plt.show()




