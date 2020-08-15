import numpy as np
import matplotlib.pyplot as plt

# %%
vlist_1 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.txt')
vlist_2 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.0543.txt')
vlist_3 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.1042.txt')
vlist_4 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.1536.txt')
vlist_5 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.2023.txt')
vlist_6 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.2506.txt')
vlist_7 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.2988.txt')
vlist_8 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.4436.txt')
vlist_9 = np.loadtxt('notebooks/SE/Kirhner/vlist_0.6924.txt')

plt.figure(dpi=300)
plt.plot(vlist_1[:, 1], vlist_1[:, 2], '.')
plt.plot(vlist_2[:, 1], vlist_2[:, 2], '.')
plt.plot(vlist_3[:, 1], vlist_3[:, 2], '.')
plt.plot(vlist_4[:, 1], vlist_4[:, 2], '.')
plt.plot(vlist_5[:, 1], vlist_5[:, 2], '.')
plt.plot(vlist_6[:, 1], vlist_6[:, 2], '.')
plt.plot(vlist_7[:, 1], vlist_7[:, 2], '.')
plt.plot(vlist_8[:, 1], vlist_8[:, 2], '.')

K_sim_120 = np.loadtxt('notebooks/SE/Kirhner/K_exp_0.txt')
plt.plot(K_sim_120[:, 0], K_sim_120[:, 1], 'o')

plt.gca().set_aspect('equal', adjustable='box')

plt.show()
