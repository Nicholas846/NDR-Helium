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

def sort_by_lm(basis_functions, l_val, m_val):
    return [i for i, bf in enumerate(basis_functions) if bf.l == l_val and bf.m == m_val]

def embed_block(D_sub, idx, basis_functions):
    n = len(basis_functions)
    D = np.zeros((n, n))
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            D[i, j] = D_sub[a, b]
    return D

def density_matrix_s_block(F, S, basis_functions, occ_s):

    idx_s = sort_by_lm(basis_functions, 0, 0)
    eps_s, C_s = sub_eigh(F, S, idx_s)
    occ_s = np.array(occ_s)
    C_occ_s = C_s[:, :len(occ_s)]
    D_s = C_occ_s @ np.diag(occ_s) @ C_occ_s.T
    D_s_full = embed_block(D_s, idx_s, basis_functions)
    D_s_full = 0.5 * (D_s_full + D_s_full.T)  
    return D_s_full, eps_s, C_s

def density_matrix_p_block(F, S, basis_functions, occ_p):
    
    D_p_full = np.zeros_like(F)
    eps_p, C_p = {}, {}

    for m in (-1, 0, 1):
        idx_p = sort_by_lm(basis_functions, 1, m)
   
        eps_m, C_m = sub_eigh(F, S, idx_p)
        eps_p[m], C_p[m] = eps_m, C_m

        occ_m = np.array(occ_p[m] if isinstance(occ_p, dict) else occ_p)
        C_occ_m = C_m[:, :len(occ_m)]
        D_m = C_occ_m @ np.diag(occ_m) @ C_occ_m.T
        D_p_full += embed_block(D_m, idx_p, basis_functions)
        
    D_p_full = 0.5 * (D_p_full + D_p_full.T)

    return D_p_full, eps_p, C_p

def density_matrix(Ds_full, Dp_full):

    return Ds_full + Dp_full









