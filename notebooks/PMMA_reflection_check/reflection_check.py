import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %%
def get_scattered_O_matrix(O_matrix, phi, theta):
    W = np.mat([[np.cos(phi), np.sin(phi), 0.],
                [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])
    return np.matmul(W, O_matrix)


def make_simple_step(x_0, O_matrix, step_length):
    return np.matmul(O_matrix.transpose(), x_0) * step_length


# %%
x0 = np.mat([[0.], [0.], [1.]])

coords = np.zeros((5, 3))

now_coords = np.zeros(3)
coords[0, :] = now_coords

O_mat = np.mat(np.eye(3))
now_coords += np.array(make_simple_step(x0, O_mat, 1)).reshape((3, ))
coords[1, :] = now_coords

O_mat = get_scattered_O_matrix(O_mat, np.pi/6, np.pi/3)
now_coords += np.array(make_simple_step(x0, O_mat, 1)).reshape((3, ))
coords[2, :] = now_coords

O_mat[:, 2] *= -1
now_coords += np.array(make_simple_step(x0, O_mat, 1)).reshape((3, ))
coords[3, :] = now_coords

O_mat = get_scattered_O_matrix(O_mat, np.pi/6, np.pi/3)
now_coords += np.array(make_simple_step(x0, O_mat, 1)).reshape((3, ))
coords[4, :] = now_coords

# plt.figure(dpi=300)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(coords[:, 0], coords[:, 1], coords[:, 2])

plt.grid()
# plt.show()
