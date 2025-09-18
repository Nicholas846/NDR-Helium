import numpy as np
from scipy.linalg import eigh

def group_by_lm(basis_functions):

    blocks = {}

    for i, bf in enumerate(basis_functions):
        key = (bf.l, bf.m)
        blocks.setdefault(key, []).append(i)
    for k in blocks:
        blocks[k] = sorted(blocks[k])
    
    return blocks


def build_density_matrix_blocked(F, S, basis_functions, occ):
    blocks = group_by_lm(basis_functions)
    nbf = len(basis_functions)
    D = np.zeros((nbf, nbf))
    eps_all = np.zeros(nbf)
    C_all = np.zeros((nbf, nbf))

    for (l, m), indices in blocks.items():
        F_block = F[np.ix_(indices, indices)]
        S_block = S[np.ix_(indices, indices)]
        eps_block, C_block = eigh(F_block, S_block)
        occ_block = occ[indices]
        C_occ_block = C_block[:, :len(occ_block)]
        D_block = C_occ_block @ np.diag(occ_block) @ C_occ_block.T

        D[np.ix_(indices, indices)] = D_block
        eps_all[indices] = eps_block
        C_all[np.ix_(indices, indices)] = C_block

    return D, eps_all, C_all


