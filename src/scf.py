import numpy as np
from scipy.linalg import eigh
from .one_e_matrice import S_matrix, H_matrix, T_matrix, V_matrix
from .eri import build_eri_tensor
from .JK_matrice import build_JK
from .basis_group import group_basis_by_lm
from .Density_occ import build_density, fractional_occupations


def scf(basis, Z, N_elec, max_iter=150, conv=1e-7, damping=0.3):
    n = len(basis)
    # One-electron stuff
    S = S_matrix(basis)
    H_core = H_matrix(basis, Z)

    T_mat = T_matrix(basis)
    V_ext_mat = V_matrix(basis, Z)

    # Build two-electron integrals once
    eri = build_eri_tensor(basis)

    l_values, radial_indices, lm_indices = group_basis_by_lm(basis)

    S_radial = {}
    H_radial = {}
    for l in l_values:
        idx_rad = radial_indices[l]
        S_radial[l] = S[np.ix_(idx_rad, idx_rad)]
        H_radial[l] = H_core[np.ix_(idx_rad, idx_rad)]

    eps_l = {}
    C_l = {}
    for l in l_values:
        F_l0 = H_radial[l]
        S_l = S_radial[l]
        eps_l[l], C_l[l] = eigh(F_l0, S_l)

    C_full = np.zeros((n, n))
    eps_full = np.zeros(n)

    col = 0
    for l in l_values:
        n_rad = C_l[l].shape[0]
        for a in range(n_rad):
            for m in range(-l, l + 1):
                idx_lm = lm_indices[(l, m)]
                C_full[idx_lm, col] = C_l[l][:, a]
                eps_full[col] = eps_l[l][a]
                col += 1

    occ = fractional_occupations(eps_full, N_elec)
    D = build_density(C_full, occ)

    E_old = 0.0

    for it in range(1, max_iter + 1):
        # Build Coulomb / Exchange
        J, K = build_JK(eri, D)

        # Build Fock
        F = H_core + J + K

        # Diagonalize per l
        eps_l = {}
        C_l = {}
        for l in l_values:
            idx_rad = radial_indices[l]
            F_l = F[np.ix_(idx_rad, idx_rad)]
            S_l = S_radial[l]
            eps_l[l], C_l[l] = eigh(F_l, S_l)

        C_full = np.zeros((n, n))
        eps_full = np.zeros(n)

        col = 0
        for l in l_values:
            n_rad = C_l[l].shape[0]
            for a in range(n_rad):
                for m in range(-l, l + 1):
                    idx_lm = lm_indices[(l, m)]
                    C_full[idx_lm, col] = C_l[l][:, a]
                    eps_full[col] = eps_l[l][a]
                    col += 1

        occ = fractional_occupations(eps_full, N_elec)
        D_new = build_density(C_full, occ)
        D_new = (1 - damping) * D + damping * D_new


        J, K = build_JK(eri, D_new)
        E_one = np.trace(D_new @ H_core)
        E_two = 0.5 * np.trace(D_new @ (J + K))
        E_tot = E_one + E_two

        dE = abs(E_tot - E_old)
        delta_D = np.linalg.norm(D_new - D)
        print(
            f"Iteration {it:2d}: E = {E_tot:.10f}  dE = {dE:.3e}  dD = {delta_D:.3e}"
        )

        if dE < conv and delta_D < 1e-6:
            D = D_new
            break

        E_old = E_tot
        D = D_new

    print("number of electron ", np.trace(D @ S))


    T_tot = np.trace(D @ T_mat)

    V_ext = np.trace(D @ V_ext_mat)

    V_ee = E_two

    V_tot = V_ext + V_ee

    virial_ratio = V_tot / T_tot

    print(f"Final Virial Ratio (V/T): {virial_ratio:.8f}")
    print("one electron energy", E_one)
    print("two electron energy", E_two)

    
    return {
        "E_total": E_tot,
        "E_one": E_one,
        "E_two": E_two,
        "orb_energies": eps_full,
        "density": D,
        "S": S,
        "H_core": H_core,
        "Fock": F,
        "iterations": it,
        "coefficients": C_full,
        "eri": eri,
        "basis": basis,
    }