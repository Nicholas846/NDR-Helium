import numpy as np
from .one_e_integral import overlap_integral, potential_integral, kinetic_integral


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
