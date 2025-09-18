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

def sub_eigh(F, S, idx):

    F_sub = F[np.ix_(idx, idx)]
    S_sub = S[np.ix_(idx, idx)]
    eps, C = eigh(F_sub, S_sub)
    return eps, C

def sort_by_l(basis_functions, l_val):
    return [i for i, bf in enumerate(basis_functions) if bf.l == l_val]

def embed_block(D_sub, idx, basis_functions):
    n = len(basis_functions)
    D = np.zeros((n, n))
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            D[i, j] = D_sub[a, b]
    return D

def density_matrix_s_block(F, S, basis_functions, occ_s):

    idx_s = sort_by_l(basis_functions, 0)
    eps_s, C_s = sub_eigh(F, S, idx_s)
    occ_s = np.array(occ_s)
    C_occ_s = C_s[:, :len(occ_s)]
    D_s = C_occ_s @ np.diag(occ_s) @ C_occ_s.T
    D_s_full = embed_block(D_s, idx_s, basis_functions)

    return D_s_full, eps_s, C_s

def density_matrix_p_block(F, S, basis_functions, occ_p):

    idx_p = sort_by_l(basis_functions, 1)
    eps_p, C_p = sub_eigh(F, S, idx_p)
    occ_p = np.array(occ_p)
    C_occ_p = C_p[:, :len(occ_p)]
    D_p = C_occ_p @ np.diag(occ_p) @ C_occ_p.T
    D_p_full = embed_block(D_p, idx_p, basis_functions)

    return D_p_full, eps_p, C_p

def density_matrix(Ds_full, Dp_full):

    return Ds_full + Dp_full









