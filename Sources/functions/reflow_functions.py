import numpy as np


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


def get_viscosity_experiment_Mn(T_C, Mn, power):  # aho2008.pdf, bueche1955.pdf - ???????????
    Mn_0 = 271374
    eta_pre = get_viscosity_experiment_const_M(T_C)
    eta_final = eta_pre * (Mn / Mn_0)**power
    return eta_final
