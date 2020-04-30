# %%
import importlib
from collections import deque

import numpy as np

import Electron as Electron
import Structure as Structure
import arrays as a
import constants as c
import grid as g
import utilities as u

import copy

c = importlib.reload(c)
g = importlib.reload(g)
u = importlib.reload(u)
Electron = importlib.reload(Electron)
Structure = importlib.reload(Structure)


# %%
e1 = Electron.Electron(
            e_id=0,
            parent_e_id=-1,
            E=100,
            coords=np.mat([[0.], [0.], [0.]]),
            O_matrix=np.mat(np.eye(3))
        )

e2 = Electron.Electron(
            e_id=0,
            parent_e_id=-1,
            E=100,
            coords=copy.deepcopy(e1.get_coords_matrix()),
            O_matrix=np.mat(np.eye(3))
        )


# %%
class MyClass:

    def __init__(self):
        self.array = np.mat([[0.], [0.], [0.]])

    def increase_array(self):
        self.array += np.mat([[1.], [1.], [1.]])

    def get_array(self):
        return self.array


c1 = MyClass()
c2 = MyClass()

