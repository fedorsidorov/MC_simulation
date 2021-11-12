import importlib
import numpy as np
import matplotlib.pyplot as plt
from mapping import mapping_harris as mm
mm = importlib.reload(mm)
plt.style.use('science')

# %%
grun = np.loadtxt('notebooks/G_value/curves/Grun_10keV.txt')

plt.figure(dpi=300)
plt.semilogy(grun[:, 0], grun[:, 1], '.')

plt.grid()

plt.show()


# %%
def l(f):
    return 0.74 + 4.7 * f - 8.9 * f**2 + 3.5 * f**3


zz = np.linspace(0, 3, 100)
Rg = 2.15

ee = 100e-6 * 10e+3 / 1.6e-19 / (Rg * 1e-4) * l(zz / Rg)

plt.figure(dpi=300)
# plt.semilogy(zz, ee)
plt.plot(zz, ee)

plt.xlim(0, 3)

plt.xlabel('z, um')
plt.ylabel('$\\varepsilon$, eV/cm$^3$')

plt.grid()
plt.show()

# %%
dE_dS = np.trapz(grun[:26, 1], x=grun[:26, 0] * 1e-4)
dE_500nm = dE_dS * mm.area_cm2

# %%
dE_matrix = np.load('data/e_matrix_E_dep.npy')

