# %%
import importlib

import numpy as np

from _outdated import MC_classes_cm

MC_classes = importlib.reload(MC_classes_cm)


# %%


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

