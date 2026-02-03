import numpy as np
from math import factorial
from scipy.special import gamma
from sympy.physics.wigner import wigner_3j



def Radial_repulsion_integral(bi, bj, bk, bl, L):
    a = bi.l + bj.l + 2
    b = bk.l + bl.l + 2
    p = bi.zeta + bj.zeta
    q = bk.zeta + bl.zeta

    n1 = a + L

    prefactor_1 = factorial(n1) / (p ** (n1 + 1))

    term0_1 = gamma(b - L) / (q ** (b - L))

    k_vals_1 = np.arange(0, n1 + 1, dtype=int)

    sum_terms_1 = 0.0
    for k in k_vals_1:
        coeff = (p ** k) / factorial(k)
        gam = gamma(b - L + k)
        denom = (p + q) ** (b - L + k)
        sum_terms_1 += coeff * gam / denom

    I_1 = prefactor_1 * (term0_1 - sum_terms_1)

    n2 = a - L - 1

    prefactor_2 = factorial(n2) / (p ** (a - L))

    k_vals_2 = np.arange(0, n2 + 1, dtype=int)

    sum_terms_2 = 0.0
    for k in k_vals_2:
        coeff = (p ** k) / factorial(k)
        gam = gamma(b + L + k + 1)
        denom = (p + q) ** (b + L + k + 1)
        sum_terms_2 += coeff * gam / denom

    I_2 = prefactor_2 * sum_terms_2

    return I_1 + I_2


def gaunt_matrix_element(li, mi, L, M, lj, mj):

    if (-mi + M + mj) != 0:
        return 0.0

    phase = (-1)**mi

    pref = np.sqrt((2*li + 1)*(2*L + 1)*(2*lj + 1) / (4*np.pi))

    return float(
        phase
        * pref
        * wigner_3j(li, L, lj, 0, 0, 0)
        * wigner_3j(li, L, lj, -mi, M, mj)
    )

def angular_part(bi, bj, bk, bl, L):

    s = 0.0
    for M in range(-L, L+1):
        a = gaunt_matrix_element(bi.l, bi.m, L, M, bj.l, bj.m)
        b = gaunt_matrix_element(bk.l, bk.m, L, M, bl.l, bl.m)
        s += a * np.conjugate(b)
    return (4*np.pi/(2*L + 1)) * float(s)


def electron_repulsion_integral(bi, bj, bk, bl):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l

    Lmin_ij, Lmax_ij = abs(li - lj), li + lj
    Lmin_kl, Lmax_kl = abs(lk - ll), lk + ll
    Lmin, Lmax = max(Lmin_ij, Lmin_kl), min(Lmax_ij, Lmax_kl)
    if Lmin > Lmax:
        return 0.0

    total = 0.0
    for L in range(Lmin, Lmax + 1):
        R_L = Radial_repulsion_integral(bi, bj, bk, bl, L)
        A_L = angular_part(bi, bj, bk, bl, L)
        total += A_L * R_L

    return float(total)