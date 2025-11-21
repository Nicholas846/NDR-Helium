import numpy as np


def build_density(C, occ):
    D = C @ np.diag(occ) @ C.T
    return D

def fractional_occupations(eps, n_electrons, degeneracy_tol=1e-8):
    n = len(eps)
    occ = np.zeros(n)

    idx_sorted = np.argsort(eps)
    eps_sorted = eps[idx_sorted]

    remaining_electrons = n_electrons

    i = 0
    while remaining_electrons > 0 and i < n:
        j = i + 1
        while j < n and abs(eps_sorted[j] - eps_sorted[i]) < degeneracy_tol:
            j += 1
        block = idx_sorted[i:j]
        block_size = len(block)
        block_occupancy = 2.0 * block_size

        if remaining_electrons >= block_occupancy:
            occ[block] = 2.0
            remaining_electrons -= block_occupancy
        else:
            frac = remaining_electrons / block_size
            occ[block] = frac
            remaining_electrons = 0.0

        i = j

    return occ
