import numpy as np
import matplotlib.pyplot as plt


# %% a)
xx_0 = np.linspace(-0.99, 0.99, 1000)
ff_a_0 = np.exp(2 * np.abs(xx_0) % 2)

plt.figure(dpi=600)
plt.plot(xx_0, ff_a_0)
plt.grid()
plt.show()


# %%
xx = np.linspace(-3, 3, 1000)

ff_a = np.ones(len(xx)) * (np.exp(2) - 1) / 2

for n in range(1, 2):
    ff_a += 4 * (np.exp(2) * (-1)**n - 1) / (np.pi**2 * n**2 + 4) * np.cos(np.pi * n * xx)

plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx_0, ff_a_0, linewidth=4, label='$f(x)$')
# plt.plot(xx, ff_a, label='$S_{2}(x)$')
# plt.plot(xx, ff_a, label='$S_{3}(x)$')
plt.plot(xx, ff_a, label='$S_{10}(x)$')

plt.legend()
plt.title('а)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $S(x)$')

plt.xlim(-3, 3)
plt.ylim(0, 8)

plt.grid()
# plt.savefig('a_2.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('a_3.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('a_10.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% b)
xx_0 = np.linspace(-0.99, 0.99, 1000)
ff_b_0 = np.exp(2 * np.abs(xx_0) % 2) * np.sign(xx_0)

plt.figure(dpi=600)
plt.plot(xx_0, ff_b_0)
plt.grid()
plt.show()


# %%
xx_0 = np.linspace(-3, 3, 1000)
xx = np.concatenate([xx_0 - 6, xx_0, xx_0 + 6])

ff_b = np.zeros(len(xx))

for n in range(1, 101):
    ff_b += 2 * np.pi * n * (1 - np.exp(2)*(-1)**n) / (np.pi**2 * n**2 + 4) * np.sin(np.pi * n * xx)

plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx / 3, np.concatenate([ff_b_0, ff_b_0, ff_b_0]), linewidth=4, label='$f(x)$')
# plt.plot(xx, ff_b, label='$S_{2}(x)$')
plt.plot(xx, ff_b, label='$S_{3}(x)$')
# plt.plot(xx, ff_b, label='$S_{10}(x)$')

plt.legend()
plt.title('б)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $S(x)$')

plt.xlim(-3, 3)
plt.ylim(-7.5, 7.5)

plt.grid()
# plt.savefig('b_2.jpg', dpi=600, bbox_inches='tight')
plt.savefig('b_3.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('b_10.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% c)
xx_0 = np.linspace(-0.99, 0.99, 1000)
ff_c_0 = np.exp(2 * np.abs(xx_0) % 2)

ff_c_0[np.where(xx_0 < 0)] = 0

plt.figure(dpi=600)
plt.plot(xx_0, ff_c_0)
plt.grid()
plt.show()


# %%
xx = np.linspace(-3, 3, 1000)

ff_c = np.ones(len(xx)) * 1/4 * (np.exp(2) - 1)

for n in range(1, 11):
    ff_c += 2 * (np.exp(2) * (-1)**n - 1) / (np.pi**2 * n**2 + 4) * np.cos(np.pi * n * xx)
    ff_c += np.pi * n * (1 - np.exp(2)*(-1)**n) / (np.pi**2 * n**2 + 4) * np.sin(np.pi * n * xx)

plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx_0, ff_c_0, linewidth=4, label='$f(x)$')
# plt.plot(xx, ff_c, label='$S_{2}(x)$')
# plt.plot(xx, ff_c, label='$S_{3}(x)$')
plt.plot(xx, ff_c, label='$S_{10}(x)$')

plt.legend()
plt.title('в)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $S(x)$')

plt.xlim(-3, 3)
plt.ylim(-2, 8)

plt.grid()
# plt.savefig('c_2.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('c_3.jpg', dpi=600, bbox_inches='tight')
plt.savefig('c_10.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% a)
xx_0 = np.linspace(-0.99, 0.99, 1000)
ff_a_0 = np.exp(2 * np.abs(xx_0) % 2)

plt.figure(dpi=600)
plt.plot(xx_0, ff_a_0)
plt.grid()
plt.show()


# %%
xx_1 = np.linspace(0, 2, 1000)
xx_2 = np.linspace(2, 4, 1000)

ff_1 = np.ones(len(xx_1)) * (1 - xx_1/2)
ff_2 = np.ones(len(xx_2)) * (-1 + xx_2/2)

xx = np.concatenate([xx_1, xx_2])
ff = np.concatenate([ff_1, ff_2])

xx_final = np.concatenate([xx - 4, xx, xx + 4])
ff_final = np.concatenate([ff, ff, ff])

SS = np.ones(len(xx_final)) * 1/2

for n in range(1, 101):
    SS += 2 * (1 - (-1)**n) / (np.pi**2 * n**2) * np.cos(np.pi * n * xx_final / 2)

plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx_final, ff_final, linewidth=4, label='$f(x)$')
plt.plot(xx_final, SS, label='$S(x)$')

plt.legend()
plt.title('а)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $S(x)$')

# plt.xlim(-3, 3)
# plt.ylim(0, 8)

plt.grid()
# plt.savefig('a_2.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('a_3.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('a_10.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %%  4 Sasha
xx_1 = np.linspace(0, np.pi, 1000)
xx_2 = np.linspace(np.pi, 2*np.pi, 1000)

ff_1 = np.ones(len(xx_1)) * 4
ff_2 = np.ones(len(xx_2)) * 1

xx = np.concatenate([xx_1, xx_2])
ff = np.concatenate([ff_1, ff_2])

xx_final = xx
ff_final = ff

# xx_final = np.concatenate([xx - 4 * np.pi, xx - 2 * np.pi, xx, xx + 2 * np.pi])
# ff_final = np.concatenate([ff, ff, ff, ff])

SS = np.ones(len(xx_final)) * 5/2

n_max = 20

for n in range(1, n_max + 1):
    SS += 3 * (1 - (-1)**n) / (np.pi * n) * np.sin(n * xx_final)

plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx_final, ff_final, linewidth=3, label='$f(x)$')
plt.plot(xx_final, SS, label='$S_{' + str(n_max) + '}(x)$')

plt.legend()
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $S(x)$')

plt.xlim(-2, 8)
# plt.xlim(-9, 9)
plt.ylim(0, 5)

plt.grid()
plt.savefig('S_' + str(n_max) + '_period.jpg', dpi=600, bbox_inches='tight')
plt.show()










