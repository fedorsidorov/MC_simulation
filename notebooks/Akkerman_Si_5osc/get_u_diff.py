import numpy as np
import matplotlib.pyplot as plt
import importlib
from scipy import integrate
from tqdm import tqdm

from functions import MC_functions as mcf
import constants as const
import grid

const = importlib.reload(const)
grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %%
E0, A0, w0, E1, A1, w1, E2, A2, w2, E3, A3, w3, E4, A4, w4 =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/fit_5osc/params_E_A_w_x5.npy')


def get_OSC(hw, E, A, w):
    return A * w * hw / ((hw**2 - E**2)**2 + (hw * w)**2)


def get_OSC_edge(hw, E, A, w):
    OLF = get_OSC(hw, E, A, w)
    OLF[np.where(hw < E)] = 0
    return OLF


def get_OSC_edge_for_int(hw, E, A, w):
    if hw < E:
        return 0
    else:
        return get_OSC(hw, E, A, w)


def get_OLF(hw):
    OLF = get_OSC(hw, E0, A0, w0)

    OLF += get_OSC_edge(hw, E1, A1, w1)
    OLF += get_OSC_edge(hw, E2, A2, w2)
    OLF += get_OSC_edge(hw, E3, A3, w3)
    OLF += get_OSC_edge(hw, E4, A4, w4)

    return OLF


def get_E_q(E, q):
    return E + const.hbar**2 * q**2 / (2 * const.m) / const.eV


def get_ELF(hw, q):
    ELF = get_OSC(hw, get_E_q(E0, q), A0, w0)

    ELF += get_OSC_edge(hw, get_E_q(E1, q), A1, w1)
    ELF += get_OSC_edge(hw, get_E_q(E2, q), A2, w2)
    ELF += get_OSC_edge(hw, get_E_q(E3, q), A3, w3)
    ELF += get_OSC_edge(hw, get_E_q(E4, q), A4, w4)

    return ELF


def get_ELF_for_int(hw, q):
    ELF = get_OSC(hw, get_E_q(E0, q), A0, w0)

    ELF += get_OSC_edge_for_int(hw, get_E_q(E1, q), A1, w1)
    ELF += get_OSC_edge_for_int(hw, get_E_q(E2, q), A2, w2)
    ELF += get_OSC_edge_for_int(hw, get_E_q(E3, q), A3, w3)
    ELF += get_OSC_edge_for_int(hw, get_E_q(E4, q), A4, w4)

    return ELF


def get_ELF_for_int_0(hw, q):
    ELF = get_OSC(hw, get_E_q(E0, q), A0, w0)
    return ELF


def get_ELF_for_int_1(hw, q):
    ELF = get_OSC_edge_for_int(hw, get_E_q(E1, q), A1, w1)
    return ELF


def get_ELF_for_int_1_no_edge(hw, q):
    ELF = get_OSC(hw, get_E_q(E1, q), A1, w1)
    return ELF


def get_ELF_for_int_2(hw, q):
    ELF = get_OSC_edge_for_int(hw, get_E_q(E2, q), A2, w2)
    return ELF


def get_ELF_for_int_2_no_edge(hw, q):
    ELF = get_OSC(hw, get_E_q(E2, q), A2, w2)
    return ELF


def get_ELF_for_int_3(hw, q):
    ELF = get_OSC_edge_for_int(hw, get_E_q(E3, q), A3, w3)
    return ELF


def get_ELF_for_int_3_no_edge(hw, q):
    ELF = get_OSC(hw, get_E_q(E3, q), A3, w3)
    return ELF


def get_ELF_for_int_4(hw, q):
    ELF = get_OSC_edge_for_int(hw, get_E_q(E4, q), A4, w4)
    return ELF


def get_ELF_for_int_4_no_edge(hw, q):
    ELF = get_OSC(hw, get_E_q(E4, q), A4, w4)
    return ELF


# %%
paper_arr = np.loadtxt(
    '/Users/fedor/PycharmProjects/MC_simulation/notebooks/Akkerman_Si_5osc/curves/ELF_q_Akkerman.txt'
)

EE = np.linspace(0, 150, 100)

plt.figure(dpi=300)

plt.plot(paper_arr[:, 0], paper_arr[:, 1], 'ro')

plt.plot(EE, get_ELF(EE, 0 * 1e+7))
plt.plot(EE, get_ELF(EE, 10 * 1e+7))
plt.plot(EE, get_ELF(EE, 20 * 1e+7))
plt.plot(EE, get_ELF(EE, 30 * 1e+7))
plt.plot(EE, get_ELF(EE, 40 * 1e+7))
plt.plot(EE, get_ELF(EE, 50 * 1e+7))

plt.show()


# %%
def get_qm_qp(E, hw):
    qm = np.sqrt(2 * const.m) / const.hbar * (np.sqrt(E) - np.sqrt(E - hw))
    qp = np.sqrt(2 * const.m) / const.hbar * (np.sqrt(E) + np.sqrt(E - hw))
    return qm, qp


def get_Si_u_diff(E_eV, hw_eV, n_osc):
    if hw_eV > E_eV:
        return 0

    E = E_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(q):
        if n_osc == 0:
            return get_ELF_for_int_0(hw_eV, q) / q
        elif n_osc == 1:
            return get_ELF_for_int_1(hw_eV, q) / q
        elif n_osc == 2:
            return get_ELF_for_int_2(hw_eV, q) / q
        elif n_osc == 3:
            return get_ELF_for_int_3(hw_eV, q) / q
        elif n_osc == 4:
            return get_ELF_for_int_4(hw_eV, q) / q

    qm, qp = get_qm_qp(E, hw)
    integral = integrate.quad(get_Y, qm, qp)[0]

    return 1 / (np.pi * const.a0 * E_eV) * integral  # cm^-1 * eV^-1


def get_Si_u_diff_no_edge(E_eV, hw_eV, n_osc):
    if hw_eV > E_eV:
        return 0

    E = E_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(q):
        if n_osc == 1:
            return get_ELF_for_int_0(hw_eV, q) / q
        elif n_osc == 2:
            return get_ELF_for_int_1_no_edge(hw_eV, q) / q
        elif n_osc == 3:
            return get_ELF_for_int_2_no_edge(hw_eV, q) / q
        elif n_osc == 4:
            return get_ELF_for_int_3_no_edge(hw_eV, q) / q
        elif n_osc == 5:
            return get_ELF_for_int_4_no_edge(hw_eV, q) / q

    qm, qp = get_qm_qp(E, hw)
    integral = integrate.quad(get_Y, qm, qp)[0]

    return 1 / (np.pi * const.a0 * E_eV) * integral  # cm^-1 * eV^-1


# %%
# for n_shell in [5]:
for n_shell in [0, 1, 2, 3, 4]:

    print(n_shell)

    u_diff = np.zeros((len(grid.EE), len(grid.EE)))

    progress_bar = tqdm(total=len(grid.EE), position=0)

    for i, now_E in enumerate(grid.EE):
        for j, now_hw in enumerate(grid.EE):
            u_diff[i, j] = get_Si_u_diff(now_E, now_hw, n_osc=n_shell)

        progress_bar.update()

    # %
    u_diff_prec = np.zeros((len(grid.EE_prec), len(grid.EE_prec)))

    progress_bar = tqdm(total=len(grid.EE_prec), position=0)

    for i, now_E in enumerate(grid.EE_prec):
        for j, now_hw in enumerate(grid.EE_prec):
            u_diff_prec[i, j] = get_Si_u_diff(now_E, now_hw, n_osc=n_shell)

        progress_bar.update()

    # np.save('notebooks/Akkerman_Si_5osc/DIIMFP_5osc_edge/u_diff_' + str(n_shell) + '.npy', u_diff)
    # np.save('notebooks/Akkerman_Si_5osc/DIIMFP_5osc_edge/u_diff_prec' + str(n_shell) + '.npy', u_diff_prec)

    # np.save('notebooks/Akkerman_Si_5osc/u_diff/u_diff_' + str(n_shell) + '.npy', u_diff)
    # np.save('notebooks/Akkerman_Si_5osc/u_diff/u_diff_prec' + str(n_shell) + '.npy', u_diff_prec)

# %%
u_test = np.zeros(len(grid.EE_prec))

for i, now_E in enumerate(grid.EE_prec):
    inds = np.where(grid.EE_prec < now_E / 2)[0]
    u_test = np.trapz(u_diff_prec[i, inds], x=grid.EE_prec[inds])






