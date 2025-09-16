import numpy as np
from scipy.linalg import eigh

def build_density_matrix(F, S, occ):
    # Solve generalized eigenproblem FC = SCE
    eps, C = eigh(F, S)
    occ = np.asarray(occ, dtype=float)
    C_occ = C[:, :len(occ)]
    D = C_occ @ np.diag(occ) @ C_occ.T
    return D, eps, C

def build_JK(D, ERI):
    # Coulomb and exchange using physicist notation ERI[i,j,k,l]
    J = np.einsum("kl,ijkl->ij", D, ERI)
    K = np.einsum("kl,ikjl->ij", D, ERI)
    return J, K

def scf_loop(basis_functions, Z, occ, S, H_core, ERI, max_iter=50, conv=1e-6):
    # Start with core Hamiltonian
    F = H_core.copy()
    D, eps, C = build_density_matrix(F, S, occ)

    E_old = 0.0
    for it in range(1, max_iter + 1):
        J, K = build_JK(D, ERI)
        F = H_core + J - 0.5 * K
        F = 0.5 * (F + F.T)  # symmetrize

        eps, C = eigh(F, S)
        D_new = C[:, :len(occ)] @ np.diag(occ) @ C[:, :len(occ)].T

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
