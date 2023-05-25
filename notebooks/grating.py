import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt

# %%
xx = np.linspace(-4, 4, 1000)

yy = np.sin(1.7*pi*xx)**2 / (1.7*pi*xx)**2 * np.sin(20.4*pi*xx)**2 / np.sin(5.1*pi*xx)**2

plt.figure(dpi=300)
plt.semilogy(xx, yy)

plt.xlabel('sin $\phi$')
plt.ylabel('I/I$_0$')
plt.grid()


plt.show()


