import numpy as np
from scipy. integrate import odeint
import matplotlib.pyplot as plt


# create function
def f(y, t):
    y1, y2 = y
    return [y2, - y2 - y1]


t = np.linspace(0, 10, 41)  # vector of time
y0 = [0, 1]  # start value
w = odeint(f, y0, t)  # solve eq.
y1 = w[:, 0]
y2 = w[:, 1]
fig = plt.figure(facecolor='white')

plt.plot(t, y1, '-o', t, y2, '-o', linewidth=2)
plt.ylabel('z')
plt.xlabel('t')
plt.grid(True)
plt.show()  # display
