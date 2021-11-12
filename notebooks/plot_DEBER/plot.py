import numpy as np
import matplotlib.pyplot as plt

# %%
pr_3 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_3d.csv', delimiter=',', skiprows=5)
xx_3 = pr_3[:, 0] / 1e+3
yy_3 = pr_3[:, 1]

plt.figure(dpi=300)
plt.plot(xx_3, yy_3, label='3')

plt.xlim(10, 20)

# plt.show()

pr_2 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/2_2.csv', delimiter=',', skiprows=5)
xx_2 = pr_2[:, 0] / 1e+3
yy_2 = pr_2[:, 1]

# plt.figure(dpi=300)
plt.plot(xx_2 - 1.3, yy_2 + 10, label='2')

# plt.xlim(10, 20)

# plt.show()

pr_1 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/1_2c.csv', delimiter=',', skiprows=5)
xx_1 = pr_1[:, 0] / 1e+3
xx_2 = pr_1[:, 1]

# plt.figure(dpi=300)
plt.plot(xx_1 + 1.1, xx_2 + 20, label='1')

plt.grid()
plt.xlim(10, 20)

plt.xlabel('x, um')
plt.ylabel('y, nm')

plt.legend()

# plt.show()
plt.savefig('profiles_123.jpg')

# %%
pr_3_1 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_1.csv', delimiter=',', skiprows=5)
xx_3_1 = pr_3[:, 0] / 1e+3
yy_3_1 = pr_3[:, 1]

pr_3_2 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_1.csv', delimiter=',', skiprows=5)
xx_3_2 = pr_3_2[:, 0] / 1e+3 + 1
yy_3_2 = pr_3_2[:, 1] + 20

pr_3_3a = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_3a.csv', delimiter=',', skiprows=5)
xx_3_3a = pr_3_3a[:, 0] / 1e+3 + 1.8
yy_3_3a = pr_3_3a[:, 1] + 30

pr_3_3b = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_3b.csv', delimiter=',', skiprows=5)
xx_3_3b = pr_3_3b[:, 0] / 1e+3 + 1.8
yy_3_3b = pr_3_3b[:, 1] + 30

pr_3_3c = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_3c.csv', delimiter=',', skiprows=5)
xx_3_3c = pr_3_3c[:, 0] / 1e+3 + 1.3
yy_3_3c = pr_3_3c[:, 1] + 40

pr_3_4a = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_4a.csv', delimiter=',', skiprows=5)
xx_3_4a = pr_3_4a[:, 0] / 1e+3 + 0.8
yy_3_4a = pr_3_4a[:, 1] + 40

pr_3_4b = np.loadtxt('notebooks/plot_DEBER/DEBER_356/3_4b.csv', delimiter=',', skiprows=5)
xx_3_4b = pr_3_4b[:, 0] / 1e+3 - 1.5
yy_3_4b = pr_3_4b[:, 1] + 40

plt.figure(dpi=300)
plt.plot(xx_3_1, yy_3_1, label='3_1')
plt.plot(xx_3_2, yy_3_2, label='3_2')
plt.plot(xx_3_3a, yy_3_3a, label='3_3a')
# plt.plot(xx_3_3b, yy_3_3b, label='3_3b')
# plt.plot(xx_3_3c, yy_3_3c, label='3_3c')
plt.plot(xx_3_4a, yy_3_4a, label='3_4a')
plt.plot(xx_3_4b, yy_3_4b, label='3_4b')

plt.xlabel('x, um')
plt.ylabel('y, nm')

plt.xlim(10, 20)
plt.grid()
plt.legend()
# plt.show()
plt.savefig('profiles_3.jpg')

# %%
pr_2_1 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/2_1.csv', delimiter=',', skiprows=5)
xx_2_1 = pr_2_1[:, 0] / 1e+3
yy_2_1 = pr_2_1[:, 1]

pr_2_2 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/2_2.csv', delimiter=',', skiprows=5)
xx_2_2 = pr_2_2[:, 0] / 1e+3 - 0.5
yy_2_2 = pr_2_2[:, 1] + 2

pr_2_3 = np.loadtxt('notebooks/plot_DEBER/DEBER_356/2_3.csv', delimiter=',', skiprows=5)
xx_2_3 = pr_2_3[:, 0] / 1e+3
yy_2_3 = pr_2_3[:, 1] + 2

plt.figure(dpi=300)
plt.plot(xx_2_1, yy_2_1, label='2_1')
plt.plot(xx_2_2, yy_2_2, label='2_2')
plt.plot(xx_2_3, yy_2_3, label='2_3')
plt.xlabel('x, um')
plt.ylabel('y, nm')
plt.xlim(10, 20)
plt.grid()
plt.legend()

# plt.show()
plt.savefig('profiles_2.jpg')


