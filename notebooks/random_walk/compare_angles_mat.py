# %%
import numpy as np
from numpy import sin, cos, arccos
import matplotlib.pyplot as plt


class VectorVect:

    def __init__(self, phi, theta):
        self.phi = phi
        self.theta = theta

    def scatter(self, phi, theta):

        # if self.theta == 0:
        #     self.theta = 1e-5

        new_theta = arccos(cos(self.theta) * cos(theta) + sin(self.theta) * sin(theta) * cos(phi))
        cos_delta_phi = (cos(theta) - cos(new_theta) * cos(self.theta)) / (sin(self.theta) * sin(new_theta))

        if cos_delta_phi < -1:
            cos_delta_phi = -1
        elif cos_delta_phi > 1:
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


# %%
angle_pairs = [[2*np.pi/i, np.pi/i] for i in range(5, 10)]

vv = VectorVect(0, 1e-15)
vm = VectorMat(np.mat(np.eye(3)))

print('START:')
print(vv.get_ort())
print(vm.get_ort())
print('__________')

for pair in angle_pairs:
    vv.scatter(-np.pi/2 - pair[0], pair[1])
    vm.scatter(pair[0], pair[1])
    print(vv.get_ort())
    print(vm.get_ort())
    print('__________')

