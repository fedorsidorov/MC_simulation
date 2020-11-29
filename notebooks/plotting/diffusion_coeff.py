import numpy as np
import importlib
import matplotlib.pyplot as plt
from functions import diffusion_functions as df

df = importlib.reload(df)

# %%
dT = np.linspace(-10, 50)

DD = df.get_D(dT, wp=1)  # in cm^2 / s

plt.figure(dpi=300)
plt.plot(dT, np.log10(df.get_D(dT, wp=0.9)), label='w$_p$ = 0.9')
plt.plot(dT, np.log10(df.get_D(dT, wp=0.95)), label='w$_p$ = 0.95')
plt.plot(dT, np.log10(df.get_D(dT, wp=1)), label='w$_p$ = 1')

plt.legend()
plt.xlabel('T - T$_g$')
plt.ylabel('log(D), cm$^2$/s')

plt.grid()
# plt.show()
plt.savefig('diffusion.png', dpi=300)

