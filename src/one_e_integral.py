
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

def S_matrix(basis_functions):
    n = len(basis_functions)
    S = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            val = overlap_integral(bi, bj)
            S[i, j] = val
            S[j, i] = val

    return 0.5 * (S + S.T)


def H_matrix(basis_functions, Z):
    n = len(basis_functions)
    H = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            tij = kinetic_integral(bi, bj)
            vij = potential_integral(bi, bj, Z)
            val = tij + vij
            H[i, j] = val
            H[j, i] = val

    return 0.5 * (H + H.T)


def T_matrix(basis_functions):
    n = len(basis_functions)
    T = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            val = kinetic_integral(bi, bj)
            T[i, j] = val
            T[j, i] = val
    return 0.5 * (T + T.T)


def V_matrix(basis_functions, Z):
    n = len(basis_functions)
    V = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            val = potential_integral(bi, bj, Z)
            V[i, j] = val
            V[j, i] = val
    return 0.5 * (V + V.T)