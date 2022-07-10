import numpy as np
from scipy.special import binom


P_H0 = 2/7 * binom(3, 2) / binom(5, 2)

P_H1 = 5/7 * binom(3, 2) / binom(5, 2) + 2/7 * binom(3, 1) * binom(2, 1) / binom(5, 2)

P_H2 = 5/7 * binom(3, 1) * binom(2, 1) / binom(5, 2) + 2/7 * binom(2, 2) / binom(5, 2)

P_H3 = 5/7 * binom(2, 2) / binom(5, 2)

P_H0 + P_H1 + P_H2 + P_H3









