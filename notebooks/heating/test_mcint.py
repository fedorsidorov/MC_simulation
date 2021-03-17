import numpy as np
import mcint
import random


# %%
def integrand(x):
    r = x[0]
    phi = x[1]
    theta = x[2]

    return r**2 * np.sin(theta)


def sampler():
    while True:
        r = random.uniform(0, 1)
        phi = random.uniform(0, 2 * np.pi)
        theta = random.uniform(0, np.pi)
        yield (r, phi, theta)


# %%
domainsize = 1 * 2 * np.pi * np.pi

result, error = mcint.integrate(integrand, sampler(), measure=domainsize, n=1000000)





