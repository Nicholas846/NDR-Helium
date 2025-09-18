import numpy as np
from scipy.linalg import eigh
from .density import density_matrix_s_block, density_matrix_p_block, density_matrix
from .matrices import S_matrix, H_matrix, ERI_tensor
from .basis import BasisFunction, make_basis

def build_JK(D, ERI):

    J = np.einsum("kl,ijkl->ij", D, ERI)
    K = np.einsum("kl,ikjl->ij", D, ERI)
    return 0.5*(J+J.T), 0.5*(K+K.T)

def scf_loop(basis_functions, Z, occ_s, occ_p, max_iter=50, conv=1e-6):

    S = S_matrix(basis_functions)
    H_core = H_matrix(basis_functions, Z)
    ERI = ERI_tensor(basis_functions)
 
    F = 0.5 * (H_core + H_core.T)  

    Ds, _, _ = density_matrix_s_block(F, S, basis_functions, occ_s)
    if occ_p.size > 0:
        Dp, _, _ = density_matrix_p_block(F, S, basis_functions, occ_p)
    else:
        Dp = np.zeros_like(Ds)
    D = density_matrix(Ds, Dp)

    E_old = 0.0
    for it in range(1, max_iter + 1):

        J, K = build_JK(D, ERI)
        F = H_core + J - 0.5 * K
        F = 0.5 * (F + F.T)

        Ds_new, _, _ = density_matrix_s_block(F, S, basis_functions, occ_s)
        if occ_p.size > 0:
            Dp_new, _, _ = density_matrix_p_block(F, S, basis_functions, occ_p)
        else:
            Dp_new = np.zeros_like(Ds_new)
        D_new = density_matrix(Ds_new, Dp_new)

        E_one = np.sum(D_new * H_core)
        E_two = 0.5 * np.sum(D_new * (J - 0.5 * K))
        E_tot = E_one + E_two

        dE = abs(E_tot - E_old)
        dD = np.linalg.norm(D_new - D)

        print(f"Iter {it:2d}: E = {E_tot:.12f}  dE = {dE:.3e}  dD = {dD:.3e}")
        if dE < conv and dD < conv:
            print("SCF Converged")
            return E_tot, E_one, E_two

        D = D_new
        E_old = E_tot

    print("SCF did not converge within max iterations.")
    return E_tot, E_one, E_two
