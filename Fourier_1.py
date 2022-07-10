import numpy as np
import matplotlib.pyplot as plt


# %%
def get_an(n):
    an = -2 / np.pi / n * np.sin(np.pi * n / 3) +\
        12 / np.pi**3 / n**3 * (3 * np.sin(np.pi * n / 3) - np.pi * n * np.cos(np.pi * n / 3))
    return an


# %%
xx_1 = np.linspace(-3, -1, 50)
xx_2 = np.linspace(-1, 1, 50)
xx_3 = np.linspace(1, 3, 50)

yy_1 = np.ones(len(xx_1))
yy_2 = 1 - xx_2**2
yy_3 = np.ones(len(xx_3))

xx_f = np.linspace(-3, 3, 1000)
yy_f = np.ones(len(xx_f)) * 8 / 9

for nn in range(1, 101):
    yy_f += get_an(nn) * np.cos(np.pi * nn * xx_f / 3)

xx = np.concatenate((xx_1, xx_2, xx_3))
yy = np.concatenate((yy_1, yy_2, yy_3))

plt.figure(dpi=300)

plt.plot(xx, yy, label='f(x)')
plt.plot(xx_f, yy_f, label='Fourier series, S$_{100}$(x)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.ylim(-2, 2)
plt.show()


# %%
sum = 0

for n in range(1, 100000):
    sum += (-1)**(n * (n-1) / 2) * np.log(1 + 1/n)

print(sum)


# %%
Q = 4e-9
eps = 7
R1 = 2e-2
R2 = 4e-2
rho = 2e-6

rr = np.linspace(0.01, 6, 1000) * 1e-2
DD = np.zeros(len(rr))

for i, r in enumerate(rr):
    if r < 4e-2:
        DD[i] = Q / (4 * np.pi * r**2) + rho * r / 3 - rho * R1**3 / (3 * r**2)
    else:
        DD[i] = (Q + (4 * rho * np.pi * (R2**3 - R1**3)) / 3) / (4 * np.pi * r**2)

DD[np.where(rr < 2e-2)] = 0

plt.figure(dpi=300)
plt.plot(rr, DD)
plt.show()




