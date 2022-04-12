import numpy as np
import matplotlib.pyplot as plt


# %%
def get_viscosity_experiment_const_M(T_C):  # aho2008.pdf
    eta_0 = 1.54e+6
    T0 = 200
    C1 = 7.6682
    C2 = 210.76
    log_aT = -C1 * (T_C - T0) / (C2 + (T_C - T0))
    eta = eta_0 * np.exp(log_aT)
    return eta


def get_SE_mobility(eta):
    C = 0.0381
    k = 0.9946
    time2scale = C * eta**k
    mobility = 1 / time2scale
    return mobility


def get_eta(mobility):
    C = 0.0381
    k = 0.9946
    eta = (1 / mobility / C)**(1/k)
    return eta


def get_viscosity_experiment_Mn(T_C, Mn, power_high, power_low, Mn_edge=42000):
    # aho2008.pdf, bueche1955.pdf - ???????????
    # Mn_0 = 271374
    Mn_0 = 271400
    # Mn_edge = 42000

    eta_pre = get_viscosity_experiment_const_M(T_C)

    if Mn > Mn_edge:
        eta_final = eta_pre * (Mn / Mn_0) ** power_high
        return eta_final
    else:
        eta_edge = eta_pre * (Mn_edge / Mn_0) ** power_high
        eta_final = eta_edge * (Mn / Mn_edge) ** power_low
        return eta_final


def move_Mn_to_mobs(Mn, T_C, power_high, power_low, Mn_edge=42000):
    eta = get_viscosity_experiment_Mn(T_C, Mn, power_high, power_low, Mn_edge)
    return get_SE_mobility(eta)


# def get_viscosity_experiment_Mw(T_C, Mw, power):  # aho2008.pdf, bueche1955.pdf - ???????????
#     Mw_0 = 670358
#     eta_pre = get_viscosity_experiment_const_M(T_C)
#     eta_final = eta_pre * (Mw / Mw_0)**power
#     return eta_final


# %%
# MM = np.logspace(4, 5, 10)
# ETA = np.zeros(len(MM))
# TT = np.linspace(80, 150, 100)
# ETA = np.zeros(len(TT))
# ETA_new = np.zeros(len(TT))

# for i in range(len(TT)):
    # ETA[i] = get_viscosity_experiment_const_M(TT[i])
    # ETA_new[i] = get_viscosity_experiment_const_M(TT[i])

# plt.figure(dpi=300)
# plt.semilogy(TT, ETA)
# plt.semilogy(TT, ETA_new)
# plt.grid()
# plt.show()

