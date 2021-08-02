import numpy as np
import matplotlib.pyplot as plt

# %%
# p_1 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/initial/1.csv', delimiter=',', skiprows=5)
p_1 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/initial/2.csv', delimiter=',', skiprows=5)
p_2 = np.loadtxt('notebooks/kolhoz_melting/blue_rect/14h_200C/4.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(p_1[:, 0], p_1[:, 1], label='initial')
plt.plot(p_2[:, 0] + 3900, p_2[:, 1] - 10, label='14 hours')

plt.grid()
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.legend()

# plt.xlim(0, 20)
# plt.ylim(0.15, 0.4)

plt.show()




