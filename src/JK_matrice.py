import numpy as np

def build_JK(eri, D, thresh=1e-14):
    n = D.shape[0]
    J = np.zeros((n, n))
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Dkl = D[k, l]
                    J[i, j] += Dkl * eri[i, j, k, l]
                    K[i, j] -= 0.5 * Dkl * eri[i, k, j, l]

    J[np.abs(J) < thresh] = 0
    K[np.abs(K) < thresh] = 0

    return J, K


def build_density(C, occ):
    D = C @ np.diag(occ) @ C.T
    return D
