import numpy as np
import matplotlib.pyplot as plt

# %%
ff = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2]) * 1e+3
# UR = np.array([7.767, 6.991, 6.214, 5.437, 4.660, 3.884, 3.107, 2.330, 1.553, 0.777]) * 1e-3
UR = np.array([12.46, 11.22, 9.97, 8.72, 7.48, 6.23, 4.99, 3.74, 2.49, 1.25]) * 1e-3

R = 1.98e+3
UC = 7

Ieff = UR / R
YC = Ieff / UC

# plt.figure(dpi=300)
# plt.plot(ff, YC, 'o--')
# plt.show()


# %%
Up = 200e-3


def get_delta_UC(f, Ux):
    if f <= 10e+3:
        return 1 + 0.075 * Up / Ux
    else:
        return 2 + 0.15 * Up / Ux


delta_UR = np.zeros(len(ff))

for i in range(len(delta_UR)):
    delta_UR[i] = get_delta_UC(ff[i], UR[i])

# %%
delta_UC = np.array([2.14, 2.14, 2.14, 2.14, 2.14, 1.14, 1.14, 1.14, 1.14, 1.14])
delta_R = 0.51

delta_YC = delta_UC + delta_UR + delta_R

# plt.figure(dpi=300)
# plt.plot(ff, delta_YC)
# plt.show()

print(delta_YC)

# %%
print(YC * delta_YC / 100)

# %%
YC_theor = 2 * np.pi * ff * 3.709e-12
YC_exp = 2 * np.pi * ff * 4.539e-12

plt.figure(dpi=300)
plt.plot(ff, YC, 'o--')
plt.plot(ff, YC_theor, 'o--')
plt.plot(ff, YC_exp, 'o--')
plt.grid()
plt.show()


