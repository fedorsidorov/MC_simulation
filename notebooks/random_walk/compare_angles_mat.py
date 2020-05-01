# %%
import numpy as np
from numpy import sin, cos, arccos
import matplotlib.pyplot as plt


class VectorVect:

    def __init__(self, phi, theta):
        self.phi = phi
        self.theta = theta

    def scatter(self, phi, theta):

        if self.theta == 0:
            print('round theta, was', self.theta)
            self.theta = 1e-10

        new_theta = arccos(cos(self.theta) * cos(theta) + sin(self.theta) * sin(theta) * cos(phi))
        cos_delta_phi = (cos(theta) - cos(new_theta) * cos(self.theta)) / (sin(self.theta) * sin(new_theta))

        if cos_delta_phi < -1:
            print('round cos_delta_phi, was', cos_delta_phi)
            cos_delta_phi = -1
        elif cos_delta_phi > 1:
            print('round cos_delta_phi, was', cos_delta_phi)
            cos_delta_phi = 1

        delta_phi = arccos(cos_delta_phi)

        if sin(theta) * sin(phi) / sin(new_theta) < 0:
            delta_phi *= -1

        new_phi = self.phi + delta_phi

        self.phi = new_phi
        self.theta = new_theta

    def get_ort(self):
        return np.array((sin(self.theta) * cos(self.phi),
                         sin(self.theta) * sin(self.phi),
                         cos(self.theta)
                         ))


# noinspection DuplicatedCode
class VectorMat:

    def __init__(self, O_matrix):
        self.x0 = np.mat([[0.], [0.], [1.]])
        self.O_matrix = O_matrix

    def scatter(self, phi, theta):
        W = np.mat([[np.cos(phi), np.sin(phi), 0.],
                    [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)],
                    [np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta), np.cos(theta)]])

        self.O_matrix = np.matmul(W, self.O_matrix)

    def get_ort(self):
        ort_mat = np.matmul(self.O_matrix.transpose(), self.x0)
        return np.array((ort_mat[0, 0], ort_mat[1, 0], ort_mat[2, 0]))


def get_angles(chain_arr):
    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return np.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        return arccos(dotproduct(v1, v2) / (length(v1) * length(v2)))

    angles = []

    for i in range(len(chain_arr) - 2):
        vector_1 = chain_arr[i + 1] - chain_arr[i]
        vector_2 = -(chain_arr[i + 2] - chain_arr[i + 1])

        angles.append(np.rad2deg(angle(vector_1, vector_2)))

    return np.array(angles)


# %%
size = 10
angle_pairs = [[np.pi/50*i, np.pi/4] for i in range(size)]

vector = VectorVect(0, 0)
# vector = VectorMat(np.mat(np.eye(3)))
coords = np.zeros((size+1, 3))
now_coords = np.array((0., 0., 0.))
coords[0, :] = now_coords

for i, pair in enumerate(angle_pairs):
    vector.scatter(pair[0], pair[1])
    now_coords += vector.get_ort()
    coords[i+1, :] = now_coords

vv_angles = get_angles(coords)
print(np.max(np.abs(vv_angles - 135)))
