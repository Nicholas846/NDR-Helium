
import numpy as np
from .integrals import overlap_integral, potential_integral, kinetic_integral, two_electron_repulsion

def S_matrix(basis_functions):
    n = len(basis_functions)
    S = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j, bj in enumerate(basis_functions):
            S[i, j] = overlap_integral(bi, bj)
    return S

def H_matrix(basis_functions, Z):
    n = len(basis_functions)
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j, bj in enumerate(basis_functions):
            T[i, j] = kinetic_integral(bi, bj)
            V[i, j] = potential_integral(bi, bj, Z)
    return T + V

'''def symmetrize_eri_chemist(ERI):
    perms = [
        (0,1,2,3),
        (1,0,2,3),
        (0,1,3,2),
        (1,0,3,2),
        (2,3,0,1),
        (3,2,0,1),
        (2,3,1,0),
        (3,2,1,0),
    ]
    return sum(ERI.transpose(p) for p in perms) / 8.0'''

def ERI_tensor(basis_functions):
    n = len(basis_functions)
    ERI = np.zeros((n, n, n, n))
    for i, bi in enumerate(basis_functions):
        for j, bj in enumerate(basis_functions):
            for k, bk in enumerate(basis_functions):
                for l, bl in enumerate(basis_functions):
                    for L in range(abs(bi.l - bj.l), bi.l + bj.l + 1):
                        ERI[i, j, k, l] += two_electron_repulsion(bi, bj, bk, bl, L)
    #ERI = symmetrize_eri_chemist(ERI)
    return ERI


