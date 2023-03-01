import numpy as np
import matplotlib.pyplot as plt


# %% pho = Z / 4 pi r d^2 hi / dr^2

Z = 80  # Hg
a0 = 5.291772e-9  # cm
file_DF = np.loadtxt('notebooks/elastic/z_080.den')
rr_DF, rho_DF = file_DF[:, 0] * a0, file_DF[:, 1] / a0 ** 3
rr = rr_DF
# rr = np.logspace(-4, 1, 1000) * a0

# %% TFM
A1, A2, A3 = 0.1, 0.55, 0.35
a1, a2, a3 = 6, 1.2, 0.3

b = 0.88534 * a0 * Z**(-1/3)

hi_TFM = A1 * np.exp(-a1 * rr / b) + A2 * np.exp(-a2 * rr / b) + A3 * np.exp(-a3 * rr / b)
rho_TFM = Z / (4 * np.pi * rr) *\
    (
            A1 * (a1 / b)**2 * np.exp(-a1 * rr / b) +
            A2 * (a2 / b)**2 * np.exp(-a2 * rr / b) +
            A3 * (a3 / b)**2 * np.exp(-a3 * rr / b)
    )

# %% TFD
x = np.log(Z)

B1 = 0.0126671 - 0.0261047 * x + 0.0214184 * x**2 - 0.00235686*x**3 + 0.0000210672 * x**4
B2 = 0.0580612 + 0.0293077 * x + 0.0857135 * x**2 - 0.0223342 * x**3 + 0.00164675 * x**4
B3 = 0.927968 - 0.00164643 * x - 0.107685 * x**2 + 0.0247998 * x**3 - 0.00167822 * x**4

b1 = (164.564 - 152.192 * x + 62.3879 * x**2 - 11.5005 * x**3 + 0.808424 * x**4) / a0
b2 = (11.3060 - 6.31902 * x + 2.26025 * x**2 - 0.370738 * x**3 + 0.0261151 * x**4) / a0
b3 = (1.48219 - 0.0557601 * x + 0.0164387 * x**2 - 0.00439703 * x**3 + 0.000997225 * x**4) / a0

hi_TFD = B1 * np.exp(-b1 * rr) + B2 * np.exp(-b2 * rr) + B3 * np.exp(-b3 * rr)

rho_TFD = Z / (4 * np.pi * rr) *\
    (
            B1 * b1**2 * np.exp(-b1 * rr) +
            B2 * b2**2 * np.exp(-b2 * rr) +
            B3 * b3**2 * np.exp(-b3 * rr)
    )

# %% DHFS
D1 = 0.2098
D2 = 0.6004
D3 = 1 - D1 - D2

d1 = 24.408
d2 = 3.9643
d3 = 1.5343

rho_DHFS = Z / (4 * np.pi * rr) *\
    (
            D1 * d1**2 / a0**2 * np.exp(-d1 * rr / a0) +
            D2 * d2**2 / a0**2 * np.exp(-d2 * rr / a0) +
            D3 * d3**2 / a0**2 * np.exp(-d3 * rr / a0)
    )

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogx(rr / a0, 4 * np.pi * rr**2 * rho_TFM * a0, label=r'ТФМ')
    ax.semilogx(rr / a0, 4 * np.pi * rr ** 2 * rho_TFD * a0, label=r'ТФД')
    ax.semilogx(rr / a0, 4 * np.pi * rr ** 2 * rho_DHFS * a0, label=r'ДХФС', color='C3')
    ax.semilogx(rr / a0, 4 * np.pi * rr ** 2 * rho_DF * a0, label=r'ДФ', color='C2')

    ax.legend(loc=1, fontsize=6)
    # ax.text(0.00015, 200 * 0.9, r'а)')
    ax.text(0.023, 200 * 0.92, r'(а)')
    ax.set(xlabel=r'$r$, $a_0$')
    ax.set(ylabel=r'$4 \pi r^2 \rho_e (r)$, а.е.')
    ax.autoscale(tight=True)

    plt.xlim(1e-4, 1e+1)
    plt.ylim(0, 200)

    fig.savefig('review_figures/e_density_models_FINAL.jpg', dpi=600)
    plt.show()


# %%
def get_phi(rho):
    phi_n_r = np.ones(len(rr)) * Z
    phi_e_r = np.zeros(len(rr))

    for i, r in enumerate(rr):
        inds = np.where(rr <= r)[0]
        phi_e_r[i] -= np.trapz(rho[inds] * 4 * np.pi * (rr[inds]) ** 2, x=rr[inds])
        phi_e_r[i] -= rr[i] * np.trapz(rho[inds[-1]:] * 4 * np.pi * rr[inds[-1]:], x=rr[inds[-1]:])

    return phi_n_r + phi_e_r


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.semilogx(rr / a0, get_phi(rho_TFM), label=r'ТФМ')
    ax.semilogx(rr / a0, get_phi(rho_TFD), label=r'ТФД')
    ax.semilogx(rr / a0, get_phi(rho_DHFS), label=r'ДХФС', color='C3')
    ax.semilogx(rr / a0, get_phi(rho_DF), label=r'ДФ', color='C2')

    ax.legend(loc=1, fontsize=6)
    # ax.text(0.00015, 100 * 0.9, r'б)')
    ax.text(0.023, 100 * 0.92, r'(б)')
    ax.set(xlabel=r'$r$, $a_0$')
    ax.set(ylabel=r'$r \varphi (r)$, а.е.')
    ax.autoscale(tight=True)

    plt.xlim(1e-4, 1e+1)
    plt.ylim(0, 100)

    fig.savefig('review_figures/phi_models_FINAL.jpg', dpi=600)
    plt.show()
