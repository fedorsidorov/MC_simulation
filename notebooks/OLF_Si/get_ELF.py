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
E1, A1, w1, E2, A2, w2, E3, A3, w3, E4, A4, w4, E5, A5, w5 =\
    np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/fit_5osc/params_E_A_w_x5.npy')


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
    OLF = get_OSC(hw, E1, A1, w1)

    OLF += get_OSC_edge(hw, E2, A2, w2)
    OLF += get_OSC_edge(hw, E3, A3, w3)
    OLF += get_OSC_edge(hw, E4, A4, w4)
    OLF += get_OSC_edge(hw, E5, A5, w5)

    return OLF


def get_E_q(E, q):
    return E + const.hbar**2 * q**2 / (2 * const.m) / const.eV


def get_ELF(hw, q):
    ELF = get_OSC(hw, get_E_q(E1, q), A1, w1)

    ELF += get_OSC_edge(hw, get_E_q(E2, q), A2, w2)
    ELF += get_OSC_edge(hw, get_E_q(E3, q), A3, w3)
    ELF += get_OSC_edge(hw, get_E_q(E4, q), A4, w4)
    ELF += get_OSC_edge(hw, get_E_q(E5, q), A5, w5)

    return ELF


def get_ELF_for_int(hw, q):
    ELF = get_OSC(hw, get_E_q(E1, q), A1, w1)

    ELF += get_OSC_edge_for_int(hw, get_E_q(E2, q), A2, w2)
    ELF += get_OSC_edge_for_int(hw, get_E_q(E3, q), A3, w3)
    ELF += get_OSC_edge_for_int(hw, get_E_q(E4, q), A4, w4)
    ELF += get_OSC_edge_for_int(hw, get_E_q(E5, q), A5, w5)

    return ELF


def get_ELF_for_int_1(hw, q):
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


def get_ELF_for_int_5(hw, q):
    ELF = get_OSC_edge_for_int(hw, get_E_q(E5, q), A5, w5)
    return ELF


def get_ELF_for_int_5_no_edge(hw, q):
    ELF = get_OSC(hw, get_E_q(E5, q), A5, w5)
    return ELF


# %%
paper_arr = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/OLF_Si/curves/ELF_q_Akkerman.txt')

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


def get_Si_DIIMFP(E_eV, hw_eV, n_shell):
    if hw_eV > E_eV:
        return 0

    E = E_eV * const.eV
    hw = hw_eV * const.eV

    def get_Y(q):
        if n_shell == 1:
            return get_ELF_for_int_1(hw_eV, q) / q
        elif n_shell == 2:
            return get_ELF_for_int_2_no_edge(hw_eV, q) / q
        elif n_shell == 3:
            return get_ELF_for_int_3_no_edge(hw_eV, q) / q
        elif n_shell == 4:
            return get_ELF_for_int_4_no_edge(hw_eV, q) / q
        elif n_shell == 5:
            return get_ELF_for_int_5_no_edge(hw_eV, q) / q

    qm, qp = get_qm_qp(E, hw)
    integral = integrate.quad(get_Y, qm, qp)[0]

    return 1 / (np.pi * const.a0 * E_eV) * integral  # cm^-1 * eV^-1


# %%
for n_shell in [1, 2, 3, 4, 5]:
# for n_shell in [5]:

    print(n_shell)

    DIIMFP = np.zeros((len(grid.EE), len(grid.EE)))

    progress_bar = tqdm(total=len(grid.EE), position=0)

    for i, now_E in enumerate(grid.EE):
        for j, now_hw in enumerate(grid.EE):
            DIIMFP[i, j] = get_Si_DIIMFP(now_E, now_hw, n_shell=n_shell)

        progress_bar.update()

    # %
    DIIMFP_prec = np.zeros((len(grid.EE_prec), len(grid.EE_prec)))

    progress_bar = tqdm(total=len(grid.EE_prec), position=0)

    for i, now_E in enumerate(grid.EE_prec):
        for j, now_hw in enumerate(grid.EE_prec):
            DIIMFP_prec[i, j] = get_Si_DIIMFP(now_E, now_hw, n_shell=n_shell)

        progress_bar.update()

    np.save('notebooks/OLF_Si/DIIMFP_5osc/DIIMFP_' + str(n_shell) + '.npy', DIIMFP)
    np.save('notebooks/OLF_Si/DIIMFP_5osc/DIIMFP_prec' + str(n_shell) + '.npy', DIIMFP_prec)

# %%
u_test = np.zeros(len(grid.EE_prec))

for i, now_E in enumerate(grid.EE_prec):
    inds = np.where(grid.EE_prec < now_E / 2)[0]
    u_test = np.trapz(DIIMFP_prec[i, inds], x=grid.EE_prec[inds])






