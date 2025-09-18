import numpy as np
from scipy.linalg import eigh

def build_JK(D, ERI):

    J = np.einsum("kl,ijkl->ij", D, ERI)
    K = np.einsum("kl,ikjl->ij", D, ERI)
    return 0.5*(J+J.T), 0.5*(K+K.T)

def scf_loop(basis_functions, Z, occ, S, H_core, ERI, max_iter=50, conv=1e-6):
    # Start with core Hamiltonian
    F = H_core.copy()
    D, eps, C = build_density_matrix(F, S, occ)

    E_old = 0.0
    for it in range(1, max_iter + 1):
        J, K = build_JK(D, ERI)
        F = H_core + J - 0.5 * K
        F = 0.5 * (F + F.T) 

        eps, C = eigh(F, S)
        D_new = C[:, :len(occ)] @ np.diag(occ) @ C[:, :len(occ)].T

        E_one = np.sum(D_new * H_core)
        E_two = 0.5 * np.sum(D_new * (J - 0.5 * K))

        E_tot = E_one + E_two 
        EJ = 0.5 * np.sum(D_new * J)        # if using spin-summed D; else EJ = np.sum(D * J)
        EK = -0.25 * np.sum(D_new * K)      # if using spin-summed D; else EK = -0.5 * np.sum(D * K)
        print(f"EJ={EJ:.6f}  EK={EK:.6f}  E2e={EJ+EK:.6f}  |EK|/EJ={abs(EK)/max(EJ,1e-16):.3f}")
        print("max|J-J^T|=", np.max(np.abs(J-J.T)), "  min diag(J)=", np.min(np.diag(J)))
        print(D_new)


        dE = abs(E_tot - E_old)
        dD = np.linalg.norm(D_new - D)

        # Debug: print orbital energies and occs
        print(f"    Orbital energies (eps): {eps[:min(5,len(eps))]}")
        #print(f"    Occupations: {occ}")
        print(f"    E_one = {E_one:.12f}, E_two = {E_two:.12f}")


        print(f"Iter {it:2d}: E = {E_tot:.12f}  dE = {dE:.3e}  dD = {dD:.3e}")
        if dE < conv and dD < conv:
            print("SCF Converged")
            return E_tot, E_one, E_two

        D = D_new
        E_old = E_tot

    print("SCF did not converge within max iterations.")
    return E_tot, E_one, E_two
