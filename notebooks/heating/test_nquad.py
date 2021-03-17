import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import nquad

from functions import MC_functions as mcf
import grid

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)


# %%
def get_Y(r, phi, theta):
    return r**2 * np.sin(theta)


integral = nquad(get_Y, [
    [0, 1],
    [0,2 * np.pi],
    [0, np.pi],
])[0]




