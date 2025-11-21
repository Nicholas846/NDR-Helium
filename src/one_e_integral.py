import numpy as np
from scipy.special import gamma

def kinetic_integral(bi, bj):
    if bi.l != bj.l or bi.m != bj.m:
        return 0
    l = bi.l
    zeta_i, zeta_j = bi.zeta, bj.zeta
    a = zeta_i + zeta_j

    term1 = (zeta_i * zeta_j * gamma(2 * l + 3)) / a ** (2 * l + 3)
    term2 = (l * a * gamma(2 * l + 2)) / a ** (2 * l + 2)
    term3 = (l * (2 * l + 1) * gamma(2 * l + 1)) / a ** (2 * l + 1)

    return 0.5 * (term1 - term2 + term3)


def potential_integral(bi, bj, Z):
    if bi.l != bj.l or bi.m != bj.m:
        return 0
    l = bi.l
    zeta_i, zeta_j = bi.zeta, bj.zeta
    return -Z * gamma(2 * l + 2) / (zeta_i + zeta_j) ** (2 * l + 2)


def overlap_integral(bi, bj):
    if bi.l != bj.l or bi.m != bj.m:
        return 0
    l = bi.l
    zeta_i, zeta_j = bi.zeta, bj.zeta
    return gamma(2 * l + 3) / (zeta_i + zeta_j) ** (2 * l + 3)

