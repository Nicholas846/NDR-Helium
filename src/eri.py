import numpy as np
from .two_e_integrals import electron_repulsion_integral

def build_eri_tensor(basis_set):
    n = len(basis_set)
    eri = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    eri[i, j, k, l] = electron_repulsion_integral(
                        basis_set[i],
                        basis_set[j],
                        basis_set[k],
                        basis_set[l],
                    )

    return eri
