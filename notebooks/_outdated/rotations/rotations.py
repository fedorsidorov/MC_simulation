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


def get_scattered_flight_ort(flight_ort, phi_scat, theta_scat):
    u, v, w = flight_ort

    if w == 1:
        u_new = np.sin(theta_scat) * np.cos(phi_scat)
        v_new = np.sin(theta_scat) * np.sin(phi_scat)
        v_new = np.cos(theta_scat)

    if w == -1:
        u_new = -np.sin(theta_scat) * np.cos(phi_scat)
        v_new = -np.sin(theta_scat) * np.sin(phi_scat)
        w_new = -np.cos(theta_scat)

    else:
        u_new = u * np.cos(theta_scat) +\
            np.sin(theta_scat) / np.sqrt(1 - w**2) * (u * w * np.cos(phi_scat) - v * np.sin(phi_scat))

        v_new = v * np.cos(theta_scat) +\
            np.sin(theta_scat) / np.sqrt(1 - w**2) * (v * w * np.cos(phi_scat) + u * np.sin(phi_scat))

        w_new = w * np.cos(theta_scat) - np.sqrt(1 - w**2) * np.sin(theta_scat) * np.cos(phi_scat)

    scattered_flight_ort = np.array((u_new, v_new, w_new))

    if np.linalg.norm(scattered_flight_ort) != 1:
        scattered_flight_ort = scattered_flight_ort / np.linalg.norm(scattered_flight_ort)

    return scattered_flight_ort


def get_ort(now_angles):

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
# now_angles = 0, 0
flight_ort = np.array((0, 0, 1))

now_ort = get_scattered_flight_ort(flight_ort, np.pi/4, np.pi/4)



