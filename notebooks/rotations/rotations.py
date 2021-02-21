import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %%
def get_Ry(theta):
    mat = np.mat([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return mat


def get_Rz(phi):
    mat = np.mat([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    return mat


def get_Ry_s(theta):
    mat = sp.Matrix([
        [sp.cos(theta), 0, sp.sin(theta)],
        [0, 1, 0],
        [-sp.sin(theta), 0, sp.cos(theta)]
    ])
    return mat


def get_Rz_s(phi):
    mat = sp.Matrix([
        [sp.cos(phi), -sp.sin(phi), 0],
        [sp.sin(phi), sp.cos(phi), 0],
        [0, 0, 1]
    ])
    return mat


def get_new_ort(now_angles, rot_angles):
# def get_rot_mat(now_angles, rot_angles):
    now_phi, now_theta = now_angles
    phi_rot, theta_rot = rot_angles

    rot_mat = np.mat([
        [np.sin(theta_rot) * np.cos(phi_rot)],
        [np.sin(theta_rot) * np.sin(phi_rot)],
        [np.cos(theta_rot)]
    ])

    rot_mat = np.matmul(get_Ry(now_theta), rot_mat)
    rot_mat = np.matmul(get_Rz(now_phi), rot_mat)

    return rot_mat


def get_ort(angles):

    now_phi, now_theta = now_angles

    now_ort = np.mat([
        [np.sin(now_theta) * np.cos(now_phi)],
        [np.sin(now_theta) * np.sin(now_phi)],
        [np.cos(now_theta)]
    ])

    return now_ort


# %%
phi, theta = sp.symbols('phi, theta')
u, v, w = sp.symbols('u, v, w')

vect = sp.Matrix([[u], [v], [w]])

# vect = sp.Matrix([
#     [sp.sin(theta) * sp.cos(phi)],
#     [sp.sin(theta) * sp.sin(phi)],
#     [sp.cos(theta)]
# ])

sp.simplify(get_Ry_s(-theta) * get_Rz_s(-phi) * vect)

# %%
now_angles = 0, 0

now_ort = get_ort(now_angles)

new_ort = get_new_ort(now_angles, [np.pi/2, np.pi/4])


