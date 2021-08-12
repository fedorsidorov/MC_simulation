import numpy as np
from numpy import sin, cos, arccos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %%
class VectorVect:

    def __init__(self, phi, theta):
        self.phi = phi
        self.theta = theta

    def scatter(self, phi, theta):
        
        if self.theta == 0:
            print('round theta, was', self.theta)
            self.theta = 1e-10
        
        new_theta = arccos(cos(self.theta) * cos(theta) + sin(self.theta) * sin(theta) * cos(phi))
        # new_theta = arccos(cos(self.theta) * cos(theta) - sin(self.theta) * sin(theta) * cos(phi))
                
        if new_theta == 0:
            new_theta = 1e-10
        
        cos_delta_phi = (cos(theta) - cos(new_theta) * cos(self.theta)) / (sin(self.theta) * sin(new_theta))

        if cos_delta_phi < -1:
            print('round cos_delta_phi, was', cos_delta_phi)
            cos_delta_phi = -1
        elif cos_delta_phi > 1:
            print('round cos_delta_phi, was', cos_delta_phi)
            cos_delta_phi = 1

        delta_phi = arccos(cos_delta_phi)

        # if sin(theta) * sin(phi) / sin(new_theta) < 0:
        if sin(theta) * sin(phi) < 0:  # sin new_theta > 0 always
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


# %
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


# %
size = 10
angle_pairs = [[np.pi/50*i, np.pi/4] for i in range(size)]

vector_V = VectorVect(0, 0)
vector_M = VectorMat(np.mat(np.eye(3)))

coords_V = np.zeros((size+1, 3))
now_coords_V = np.array((0., 0., 0.))
coords_V[0, :] = now_coords_V

coords_M = np.zeros((size+1, 3))
now_coords_M = np.array((0., 0., 0.))
coords_M[0, :] = now_coords_M


# %
for i, pair in enumerate(angle_pairs):
    
    vector_V.scatter(pair[0], pair[1])
    now_coords_V += vector_V.get_ort()
    coords_V[i+1, :] = now_coords_V
    
    vector_M.scatter(pair[0], pair[1])
    now_coords_M += vector_M.get_ort()
    coords_M[i+1, :] = now_coords_M

vv_angles_V = get_angles(coords_V)
vv_angles_M = get_angles(coords_M)

# %%
print(np.max(np.abs(vv_angles_V - 135)))

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(coords_V[:, 0], coords_V[:, 1], coords_V[:, 2])
plt.plot(coords_M[:, 0], coords_M[:, 1], coords_M[:, 2])

plt.grid()
# plt.show()

# %%
vv = VectorVect(0, 0)
vm = VectorMat(np.mat(np.eye(3)))

vv.scatter(np.pi/2, np.pi/4)
vm.scatter(np.pi/2, np.pi/4)

# vv.scatter(np.pi/2, 0)
# vm.scatter(np.pi/2, 0)

print(vv.get_ort())
print(vm.get_ort())







