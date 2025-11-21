import numpy as np
from math import factorial
from scipy.special import gamma, wigner_3j



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


def angular_part(bi, bj, bk, bl, L):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l
    mi, mj, mk, ml = bi.m, bj.m, bk.m, bl.m

    a = wigner_3j(li, L, lj, 0, 0, 0)
    b = wigner_3j(lk, L, ll, 0, 0, 0)

    sum_M = 0.0
    for M in range(-L, L + 1):
        c = wigner_3j(li, L, lj, -mi, -M, mj)
        d = wigner_3j(lk, L, ll, -mk, M, ml)
        sum_M += (-1) ** M * c * d

    return float(a * b * sum_M)


def electron_repulsion_integral(bi, bj, bk, bl):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l

    Lmin_ij, Lmax_ij = abs(li - lj), li + lj
    Lmin_kl, Lmax_kl = abs(lk - ll), lk + ll
    Lmin, Lmax = max(Lmin_ij, Lmin_kl), min(Lmax_ij, Lmax_kl)
    if Lmin > Lmax:
        return 0.0

    total = 0.0
    pref = (-1) ** (bi.m + bk.m) * np.sqrt(
        (2 * li + 1) * (2 * lj + 1) * (2 * lk + 1) * (2 * ll + 1)
    )

    for L in range(Lmin, Lmax + 1):
        A_L = angular_part(bi, bj, bk, bl, L)
        R_L = Radial_repulsion_integral(bi, bj, bk, bl, L)
        total += A_L * R_L

    return float(pref * total)
