import numpy as np
from scipy.special import gamma
from sympy.physics.wigner import wigner_3j
from scipy.linalg import eigh
from math import factorial
from scipy.optimize import minimize


class BasisFunction:
    def __init__(self, zeta, l, m):
        self.zeta = float(zeta)
        self.l = int(l)
        self.m = int(m)

    def __repr__(self):
        return f"BasisFunction(zeta={self.zeta}, l={self.l}, m={self.m})"


def make_basis(*zeta_lists):
    bf = []

    for l, zetas in enumerate(zeta_lists):
        for zeta in zetas:
            for m in range(-l, l + 1):
                bf.append(BasisFunction(zeta, l, m))
    return bf


def kinetic_integral(bi, bj):
    if bi.l != bj.l or bi.m != bj.m:
        return 0
    l = bi.l
    zeta_i, zeta_j = bi.zeta, bj.zeta
    a = zeta_i + zeta_j

    term1 = (zeta_i * zeta_j * gamma(2 * l + 3)) / a ** (2 * l + 3)
    term2 = (l * a * gamma(2 * l + 2)) / a ** (2 * l + 2)
    term3 = (l * (2 * l + 1) * gamma(2 * l + 1)) / a ** (2 * l + 1)

    return 0.5 * (term1 - term2 + term3)


def potential_integral(bi, bj, Z):
    if bi.l != bj.l or bi.m != bj.m:
        return 0
    l = bi.l
    zeta_i, zeta_j = bi.zeta, bj.zeta
    return -Z * gamma(2 * l + 2) / (zeta_i + zeta_j) ** (2 * l + 2)


def overlap_integral(bi, bj):
    if bi.l != bj.l or bi.m != bj.m:
        return 0
    l = bi.l
    zeta_i, zeta_j = bi.zeta, bj.zeta
    return gamma(2 * l + 3) / (zeta_i + zeta_j) ** (2 * l + 3)


def S_matrix(basis_functions):
    n = len(basis_functions)
    S = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            val = overlap_integral(bi, bj)
            S[i, j] = val
            S[j, i] = val

    return 0.5 * (S + S.T)


def H_matrix(basis_functions, Z):
    n = len(basis_functions)
    H = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            tij = kinetic_integral(bi, bj)
            vij = potential_integral(bi, bj, Z)
            val = tij + vij
            H[i, j] = val
            H[j, i] = val

    return 0.5 * (H + H.T)


def T_matrix(basis_functions):
    n = len(basis_functions)
    T = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            val = kinetic_integral(bi, bj)
            T[i, j] = val
            T[j, i] = val
    return 0.5 * (T + T.T)


def V_matrix(basis_functions, Z):
    n = len(basis_functions)
    V = np.zeros((n, n))
    for i, bi in enumerate(basis_functions):
        for j in range(i, n):
            bj = basis_functions[j]
            val = potential_integral(bi, bj, Z)
            V[i, j] = val
            V[j, i] = val
    return 0.5 * (V + V.T)


def Radial_repulsion_integral(bi, bj, bk, bl, L):
    a = bi.l + bj.l + 2
    b = bk.l + bl.l + 2
    p = bi.zeta + bj.zeta
    q = bk.zeta + bl.zeta

    n1 = a + L

    prefactor_1 = factorial(n1) / (p ** (n1 + 1))

    term0_1 = gamma(b - L) / (q ** (b - L))

    k_vals_1 = np.arange(0, n1 + 1, dtype=int)

    sum_terms_1 = 0.0
    for k in k_vals_1:
        coeff = (p ** k) / factorial(k)
        gam = gamma(b - L + k)
        denom = (p + q) ** (b - L + k)
        sum_terms_1 += coeff * gam / denom

    I_1 = prefactor_1 * (term0_1 - sum_terms_1)

    n2 = a - L - 1

    prefactor_2 = factorial(n2) / (p ** (a - L))

    k_vals_2 = np.arange(0, n2 + 1, dtype=int)

    sum_terms_2 = 0.0
    for k in k_vals_2:
        coeff = (p ** k) / factorial(k)
        gam = gamma(b + L + k + 1)
        denom = (p + q) ** (b + L + k + 1)
        sum_terms_2 += coeff * gam / denom

    I_2 = prefactor_2 * sum_terms_2

    return I_1 + I_2


def angular_part(bi, bj, bk, bl, L):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l
    mi, mj, mk, ml = bi.m, bj.m, bk.m, bl.m

    a = wigner_3j(li, L, lj, 0, 0, 0)
    b = wigner_3j(lk, L, ll, 0, 0, 0)

    sum_M = 0.0
    for M in range(-L, L + 1):
        c = wigner_3j(li, L, lj, -mi, -M, mj)
        d = wigner_3j(lk, L, ll, -mk, M, ml)
        sum_M += (-1) ** M * c * d

    return float(a * b * sum_M)


def electron_repulsion_test(bi, bj, bk, bl):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l

    Lmin_ij, Lmax_ij = abs(li - lj), li + lj
    Lmin_kl, Lmax_kl = abs(lk - ll), lk + ll
    Lmin, Lmax = max(Lmin_ij, Lmin_kl), min(Lmax_ij, Lmax_kl)
    if Lmin > Lmax:
        return 0.0

    total = 0.0
    pref = (-1) ** (bi.m + bk.m) * np.sqrt(
        (2 * li + 1) * (2 * lj + 1) * (2 * lk + 1) * (2 * ll + 1)
    )

    for L in range(Lmin, Lmax + 1):
        A_L = angular_part(bi, bj, bk, bl, L)
        R_L = Radial_repulsion_integral(bi, bj, bk, bl, L)
        total += A_L * R_L

    return float(pref * total)


def build_eri_tensor(basis_set):
    n = len(basis_set)
    eri = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    eri[i, j, k, l] = electron_repulsion_test(
                        basis_set[i],
                        basis_set[j],
                        basis_set[k],
                        basis_set[l],
                    )

    return eri


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


def group_basis_by_lm(basis):
    lm_indices = {}
    for i, bf in enumerate(basis):
        lm_indices.setdefault((bf.l, bf.m), []).append(i)

    l_values = sorted(set(bf.l for bf in basis))

    radial_indices = {}
    for l in l_values:
        if (l, 0) not in lm_indices:
            raise ValueError(f"No m=0 function for l = {l}")
        radial_indices[l] = lm_indices[(l, 0)]

        n_rad = len(radial_indices[l])
        for m in range(-l, l + 1):
            assert len(lm_indices[(l, m)]) == n_rad

    return l_values, radial_indices, lm_indices


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


def scf(basis, Z, N_elec, max_iter=150, conv=1e-8, damping=0.3):
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
        # D_new = (1 - damping) * D + damping * D_new
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

    # --- Virial Ratio Calculation ---
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
    }


def even_tempered(alpha, beta, n):
    return [alpha * beta ** i for i in range(n)]


'''Z = 21
zeta_s = even_tempered(1.5619112684289926, 6.104061705443798, 2)
zeta_p = even_tempered(2.8361445797672773, 9.647059115315544, 1)
zeta_d = [2.0]
basis = make_basis(zeta_s, zeta_p, zeta_d)

N_elec = Z

result = scf(basis, Z, N_elec)

print("\nSCF summary:")
print(f"  Total energy        = {result['E_total']:.10f}")
print(f"  One-electron energy = {result['E_one']:.10f}")
print(f"  Two-electron energy = {result['E_two']:.10f}")
print(f"  Iterations          = {result['iterations']}")
print(f"  orbital energy      = {result['orb_energies']}")'''


def E_B(alpha_s, beta_s, alpha_p, beta_p, alpha_d, beta_d,
        n_s, n_p, n_d, scf_fn, make_basis_fn):
    # Enforce constraints: exponents > 0, scaling factors > 1
    if (alpha_s <= 0 or beta_s <= 1 or
        alpha_p <= 0 or beta_p <= 1 or
        alpha_d <= 0 or beta_d <= 1):
        return 1e6

    # 1. Build basis sets
    zeta_s = even_tempered(alpha_s, beta_s, n_s)
    zeta_p = even_tempered(alpha_p, beta_p, n_p)
    zeta_d = even_tempered(alpha_d, beta_d, n_d)

    basis = make_basis_fn(zeta_s, zeta_p, zeta_d)


    result = scf_fn(basis, 36, 36)

    return result["E_total"]

def optimize_even_tempered_B(
    scf_fn, make_basis_fn,
    n_s=4, n_p=7, n_d=2,
    x0=(0.7, 1.8, 0.5, 1.8, 2.0, 1.5),
    max_iter=50, verbose=True
):

    def objective(x):
        alpha_s, beta_s, alpha_p, beta_p, alpha_d, beta_d = x
        return E_B(alpha_s, beta_s, alpha_p, beta_p, alpha_d, beta_d,
                   n_s, n_p, n_d, scf_fn, make_basis_fn)

    bounds = [
        (1e-3, None), (1.0 + 1e-3, None),  # s
        (1e-3, None), (1.0 + 1e-3, None),  # p
        (1e-3, None), (1.0 + 1e-3, None),  # d
    ]

    if verbose:
        print("Starting optimization...")

    result = minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-6, 'disp': verbose}
    )

    if not result.success and verbose:
        print(f"Optimization failed: {result.message}")

    a_s, b_s, a_p, b_p, a_d, b_d = result.x
    E_opt = result.fun

    # Your previous tweak: add two extra s exponents at the end
    zeta_s = even_tempered(a_s, b_s, n_s) + [0.7, 0.1]
    zeta_p = even_tempered(a_p, b_p, n_p)
    zeta_d = even_tempered(a_d, b_d, n_d)

    if verbose:
        print("\n" + "="*50)
        print("FINAL OPTIMIZED PARAMETERS")
        print("="*50)
        print(f"alpha_s = {a_s:.8f}, beta_s = {b_s:.8f}")
        print(f"alpha_p = {a_p:.8f}, beta_p = {b_p:.8f}")
        print(f"alpha_d = {a_d:.8f}, beta_d = {b_d:.8f}")
        print(f"E = {E_opt:.10f} Hartree")

    return {
        "alpha_s": a_s, "beta_s": b_s,
        "alpha_p": a_p, "beta_p": b_p,
        "alpha_d": a_d, "beta_d": b_d,
        "zeta_s": zeta_s, "zeta_p": zeta_p, "zeta_d": zeta_d,
        "E": E_opt
    }


n_s = 4
n_p = 3
n_d = 1

x0 = (
    0.1,
    2.5,
    0.3,
    4.0,
    0.7,
    4.5
)

opt_B = optimize_even_tempered_B(
    scf_fn=scf,
    make_basis_fn=make_basis,
    n_s=n_s,
    n_p=n_p,
    x0=x0,
    max_iter=50,
    verbose=True,
)

print("\nBest parameters:")
print(" alpha_s =", opt_B["alpha_s"])
print(" beta_s  =", opt_B["beta_s"])
print(" alpha_p =", opt_B["alpha_p"])
print(" beta_p  =", opt_B["beta_p"])
print(" E_tot   =", opt_B["E"])

# --- Final Check ---
basis_opt = make_basis(opt_B["zeta_s"], opt_B["zeta_p"], [])
res = scf(basis_opt, Z=36, N_elec=36)

print("\nFinal SCF check:")
print(" Total energy       =", res["E_total"])
