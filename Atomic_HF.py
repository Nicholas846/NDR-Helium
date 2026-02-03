import numpy as np
from scipy.special import gamma
from sympy.physics.wigner import wigner_3j
from scipy.linalg import eigh
from math import factorial
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.integrate import quad

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
        sum_terms_1 += (coeff * gam) / denom

    I_1 = prefactor_1 * (term0_1 - sum_terms_1)

    n2 = a - L - 1

    prefactor_2 = factorial(n2) / (p ** (a - L))

    k_vals_2 = np.arange(0, n2 + 1, dtype=int)

    sum_terms_2 = 0.0
    for k in k_vals_2:
        coeff = (p ** k) / factorial(k)
        gam = gamma(b + L + k + 1)
        denom = (p + q) ** (b + L + k + 1)
        sum_terms_2 += (coeff * gam) / denom

    I_2 = prefactor_2 * sum_terms_2

    return I_1 + I_2



'''def angular_part(bi, bj, bk, bl, L):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l
    mi, mj, mk, ml = bi.m, bj.m, bk.m, bl.m

    a = wigner_3j(li, L, lj, 0, 0, 0)
    b = wigner_3j(lk, L, ll, 0, 0, 0)

    sum_M = 0.0
    for M in range(-L, L + 1):
        e = (-1) ** M
        c = wigner_3j(li, L, lj, -mi, -M, mj)
        d = wigner_3j(lk, L, ll, -mk, M, ml)
        sum_M += e * c * d

    return float(a * b * sum_M)'''

def Y_matrix_element(li, mi, L, M, lj, mj):
    """
    < li mi | Y_{L M} | lj mj >  for *complex* spherical harmonics with Condon–Shortley phase.
    This equals ∫ Y_{li mi}^*(Ω) Y_{L M}(Ω) Y_{lj mj}(Ω) dΩ.
    """
    # selection rule: -mi + M + mj = 0 is enforced by the 3j anyway, but cheap early-out:
    if (-mi + M + mj) != 0:
        return 0.0

    # (-1)^mi from conjugation: Y_{l m}^* = (-1)^m Y_{l,-m}
    phase = (-1)**mi

    pref = np.sqrt((2*li + 1)*(2*L + 1)*(2*lj + 1) / (4*np.pi))

    return float(
        phase
        * pref
        * wigner_3j(li, L, lj, 0, 0, 0)
        * wigner_3j(li, L, lj, -mi, M, mj)
    )

def angular_factor_pair_symmetric(bi, bj, bk, bl, L):
    """
    Angular part in the pair-symmetric form:
    (4π/(2L+1)) * Σ_M <i|Y_LM|j> * <k|Y_LM|l>^*
    """
    s = 0.0
    for M in range(-L, L+1):
        a = Y_matrix_element(bi.l, bi.m, L, M, bj.l, bj.m)
        b = Y_matrix_element(bk.l, bk.m, L, M, bl.l, bl.m)
        s += a * np.conjugate(b)
    return (4*np.pi/(2*L + 1)) * float(s)

def electron_repulsion_test(bi, bj, bk, bl):
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l

    Lmin_ij, Lmax_ij = abs(li - lj), li + lj
    Lmin_kl, Lmax_kl = abs(lk - ll), lk + ll
    Lmin, Lmax = max(Lmin_ij, Lmin_kl), min(Lmax_ij, Lmax_kl)
    if Lmin > Lmax:
        return 0.0

    total = 0.0
    for L in range(Lmin, Lmax + 1):
        R_L = Radial_repulsion_integral(bi, bj, bk, bl, L)
        A_L = angular_factor_pair_symmetric(bi, bj, bk, bl, L)
        total += A_L * R_L

    return float(total)

'''def electron_repulsion_test(bi, bj, bk, bl):
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

    return float(pref * total)'''


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


def build_JK(eri, D):
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

Z = 10
zeta_s = [
    93.7103745724026,
    36.31530927584123,
    17.326076791934174,
    9.20502018607238,
    5.220119627077525,
    3.0652717789292354,
    1.3075479390829232,

    0.708271178970315
]
zeta_p = [
    5.327891916884201,
    2.504061352403331,
    1.3020698500080554,
    0.6571656186989699
]
basis = make_basis(zeta_s, zeta_p)


'''Z = 5
zeta_s = [
    1.59924861e+02, 6.99117919e+01, 2.91900911e+01, 1.13076236e+01,
    4.39809087e+00, 2.97765257e+00,
    1.38361503e+00, 5.95898289e-01,
    2.51483681e-01, 1.10195734e-01
]


zeta_p = [
    6.03321862, 2.83638718, 2.26642916,
    0.9546399, 0.36127173, 0.17002352
]

basis = make_basis(zeta_s, zeta_p)
'''
N_elec = Z

'''Z = 2
N_elec = Z
zeta_s = [13.9074, 8.2187, 26.0325, 11.9249, 4.2635, 2.8357, 2.0715,]
basis = make_basis(zeta_s)'''
result = scf(basis, Z, N_elec)

print("\nSCF summary:")
print(f"  Total energy        = {result['E_total']:.10f}")
print(f"  One-electron energy = {result['E_one']:.10f}")
print(f"  Two-electron energy = {result['E_two']:.10f}")
print(f"  Iterations          = {result['iterations']}")
print(f"  orbital energy      = {np.sort(result['orb_energies'])}")
coefficients = result['coefficients']
df = pd.DataFrame(coefficients)
print(df)


'''def overlap_and_derivs(bi, bj):
    if bi.l != bj.l or bi.m != bj.m:
        return 0.0, 0.0, 0.0
    l = bi.l
    z1, z2 = bi.zeta, bj.zeta
    a = z1 + z2
    g3 = gamma(2*l + 3)

    s = g3 / a**(2*l + 3)
    ds_da = -(2*l + 3) * g3 / a**(2*l + 4)
    ds_dz1 = ds_da
    ds_dz2 = ds_da
    return s, ds_dz1, ds_dz2


def potential_and_derivs(bi, bj, Z):
    if bi.l != bj.l or bi.m != bj.m:
        return 0.0, 0.0, 0.0
    l = bi.l
    z1, z2 = bi.zeta, bj.zeta
    a = z1 + z2
    g2 = gamma(2*l + 2)

    v = -Z * g2 / a**(2*l + 2)
    dv_da =  Z * (2*l + 2) * g2 / a**(2*l + 3)
    dv_dz1 = dv_da
    dv_dz2 = dv_da
    return v, dv_dz1, dv_dz2


def kinetic_and_derivs(bi, bj):
    if bi.l != bj.l or bi.m != bj.m:
        return 0.0, 0.0, 0.0

    l = bi.l
    z1, z2 = bi.zeta, bj.zeta
    a = z1 + z2

    g1 = gamma(2*l + 3)
    g2 = gamma(2*l + 2)
    g3 = gamma(2*l + 1)

    term1 = z1 * z2 * g1 / a**(2*l + 3)
    term2 = l * g2 * a**(-(2*l + 1))
    term3 = l * (2*l + 1) * g3 / a**(2*l + 1)
    t = 0.5 * (term1 - term2 + term3)

    n1 = 2*l + 3

    dT1_dz1 = z2 * g1 / a**n1 - n1 * z1 * z2 * g1 / a**(n1 + 1)
    dT1_dz2 = z1 * g1 / a**n1 - n1 * z1 * z2 * g1 / a**(n1 + 1)

    dT2_da = l * g2 * (-(2*l + 1)) * a**(-(2*l + 2))
    dT2_dz1 = dT2_da
    dT2_dz2 = dT2_da

    dT3_da = l * (2*l + 1) * g3 * (-(2*l + 1)) * a**(-(2*l + 2))
    dT3_dz1 = dT3_da
    dT3_dz2 = dT3_da

    dt_dz1 = 0.5 * (dT1_dz1 - dT2_dz1 + dT3_dz1)
    dt_dz2 = 0.5 * (dT1_dz2 - dT2_dz2 + dT3_dz2)

    return t, dt_dz1, dt_dz2

def radial_repulsion_and_derivs(bi, bj, bk, bl, L):
    a = bi.l + bj.l + 2
    b = bk.l + bl.l + 2
    p = bi.zeta + bj.zeta
    q = bk.zeta + bl.zeta


    n1 = a + L
    pref1 = factorial(n1) / p**(n1 + 1)
    term0_1 = gamma(b - L) / q**(b - L)

    sum1 = 0.0
    dsum1_dp = 0.0
    dsum1_dq = 0.0
    for k in range(0, n1 + 1):
        nu = b - L + k
        base = (p**k / factorial(k)) * gamma(b - L + k) / (p + q)**nu
        sum1 += base
        if p != 0.0:
            dbase_dp = base * (k/p - nu/(p+q))
        else:
            dbase_dp = -base * nu/(p+q)
        dbase_dq = base * (-nu/(p+q))
        dsum1_dp += dbase_dp
        dsum1_dq += dbase_dq

    I1 = pref1 * (term0_1 - sum1)
    dpref1_dp = pref1 * (-(n1+1)/p)
    dterm0_1_dq = term0_1 * (-(b - L)/q)
    dI1_dp = dpref1_dp * (term0_1 - sum1) - pref1 * dsum1_dp
    dI1_dq = pref1 * (dterm0_1_dq - dsum1_dq)


    n2 = a - L - 1
    pref2 = factorial(n2) / p**(a - L)

    sum2 = 0.0
    dsum2_dp = 0.0
    dsum2_dq = 0.0
    for k in range(0, n2 + 1):
        nu = b + L + k + 1
        base = (p**k / factorial(k)) * gamma(b + L + k + 1) / (p + q)**nu
        sum2 += base
        if p != 0.0:
            dbase_dp = base * (k/p - nu/(p+q))
        else:
            dbase_dp = -base * nu/(p+q)
        dbase_dq = base * (-nu/(p+q))
        dsum2_dp += dbase_dp
        dsum2_dq += dbase_dq

    I2 = pref2 * sum2
    dpref2_dp = pref2 * (-(a - L)/p)
    dI2_dp = dpref2_dp * sum2 + pref2 * dsum2_dp
    dI2_dq = pref2 * dsum2_dq

    R = I1 + I2
    dR_dp = dI1_dp + dI2_dp
    dR_dq = dI1_dq + dI2_dq
    return R, dR_dp, dR_dq


def build_param_maps(basis):

    l_values, radial_indices, lm_indices = group_basis_by_lm(basis)
    n = len(basis)
    ao_to_param = np.full(n, -1, dtype=int)
    param_to_aos = []

    p_index = 0
    for l in l_values:
        n_rad = len(radial_indices[l])
        for a in range(n_rad):
            aos = []
            for m in range(-l, l + 1):
                idx_list = lm_indices[(l, m)]
                mu = idx_list[a]   # AO index for this radial shell and m
                aos.append(mu)
                ao_to_param[mu] = p_index
            param_to_aos.append(aos)
            p_index += 1

    return l_values, radial_indices, lm_indices, param_to_aos, ao_to_param


def eri_deriv_wrt_param(mu, nu, kappa, lam, p_index, basis, ao_to_param):
    bi, bj, bk, bl = basis[mu], basis[nu], basis[kappa], basis[lam]
    li, lj, lk, ll = bi.l, bj.l, bk.l, bl.l

    Lmin_ij, Lmax_ij = abs(li - lj), li + lj
    Lmin_kl, Lmax_kl = abs(lk - ll), lk + ll
    Lmin, Lmax = max(Lmin_ij, Lmin_kl), min(Lmax_ij, Lmax_kl)
    if Lmin > Lmax:
        return 0.0

    pref = (-1)**(bi.m + bk.m) * np.sqrt(
        (2*li+1)*(2*lj+1)*(2*lk+1)*(2*ll+1)
    )

    dtotal = 0.0
    for L in range(Lmin, Lmax+1):
        A_L = angular_part(bi, bj, bk, bl, L)
        _, dR_dp, dR_dq = radial_repulsion_and_derivs(bi, bj, bk, bl, L)

        dR_dz = 0.0
        if ao_to_param[mu] == p_index:
            dR_dz += dR_dp
        if ao_to_param[nu] == p_index:
            dR_dz += dR_dp
        if ao_to_param[kappa] == p_index:
            dR_dz += dR_dq
        if ao_to_param[lam] == p_index:
            dR_dz += dR_dq

        dtotal += A_L * dR_dz

    return float(pref * dtotal)



def energy_grad_wrt_param_zeta(param_index, basis, D, W, Z, eri,
                               param_to_aos, ao_to_param):
    n = len(basis)
    dE = 0.0

    aos_p = set(param_to_aos[param_index])

    # --- one-electron part ---
    for mu in range(n):
        for nu in range(n):
            uses_mu = mu in aos_p
            uses_nu = nu in aos_p
            if not (uses_mu or uses_nu):
                continue

            bi, bj = basis[mu], basis[nu]
            t, dt_dz_mu, dt_dz_nu = kinetic_and_derivs(bi, bj)
            v, dv_dz_mu, dv_dz_nu = potential_and_derivs(bi, bj, Z)
            s, ds_dz_mu, ds_dz_nu = overlap_and_derivs(bi, bj)

            dh = 0.0
            dS = 0.0
            if uses_mu:
                dh += dt_dz_mu + dv_dz_mu
                dS += ds_dz_mu
            if uses_nu:
                dh += dt_dz_nu + dv_dz_nu
                dS += ds_dz_nu

            dE += D[mu, nu] * dh - W[mu, nu] * dS

    # --- two-electron part ---
    for mu in range(n):
        for nu in range(n):
            for kappa in range(n):
                for lam in range(n):
                    if not (
                        ao_to_param[mu] == param_index
                        or ao_to_param[nu] == param_index
                        or ao_to_param[kappa] == param_index
                        or ao_to_param[lam] == param_index
                    ):
                        continue
                    dERI = eri_deriv_wrt_param(mu, nu, kappa, lam,
                                               param_index, basis, ao_to_param)
                    Gamma = (
                        D[mu, nu] * D[kappa, lam]
                        - 0.5 * D[mu, kappa] * D[nu, lam]
                    )
                    dE += 0.5 * Gamma * dERI

    return dE



def pack_params(zeta_s, zeta_p):

    zs = np.array(zeta_s, float)
    zp = np.array(zeta_p, float)
    return np.log(np.concatenate([zs, zp]))


def unpack_params(x, n_s, n_p):
    x = np.array(x, float)
    zetas = np.exp(x)
    zeta_s = zetas[:n_s]
    zeta_p = zetas[n_s:n_s + n_p]
    return zeta_s, zeta_p





def energy_and_grad(x, n_s, n_p):
    zeta_s, zeta_p = unpack_params(x, n_s, n_p)
    basis = make_basis(zeta_s, zeta_p)
    result = scf(basis, Z, N_elec, max_iter=100, conv=1e-7, damping=0.3)

    E = result["E_total"]
    D = result["density"]
    F = result["Fock"]
    eri = result["eri"]
    basis = result["basis"]
    # in energy_and_grad, after SCF:
    eps = result["orb_energies"]
    C   = result["coefficients"]
    occ = fractional_occupations(eps, N_elec)  # same function as SCF

    # Build energy-weighted density W_{μν} = sum_i f_i ε_i C_{μi} C_{νi}
    W = C @ np.diag(eps * occ) @ C.T
    # parameter mapping
    _, _, _, param_to_aos, ao_to_param = build_param_maps(basis)
    n_params = len(param_to_aos)

    g_zeta = np.zeros(n_params)
    for p in range(n_params):
        g_zeta[p] = energy_grad_wrt_param_zeta(
            p, basis, D, W, Z, eri, param_to_aos, ao_to_param
        )

    # chain rule: dE/d(log ζ) = ζ * dE/dζ
    zetas_all = np.concatenate([zeta_s, zeta_p])
    grad_x = g_zeta * zetas_all

    return E, grad_x


# wrappers for scipy.optimize.minimize
def objective(x, n_s, n_p):
    E, _ = energy_and_grad(x, n_s, n_p)
    return E


def gradient(x, n_s, n_p):
    _, g = energy_and_grad(x, n_s, n_p)
    return g

def quasi_newton_bfgs(
    x_full0,
    n_s,
    n_p,
    free_idx,
    max_iter=50,
    gtol=1e-5,
    step0=1.0,
    c1=1e-4,
    z_min=1e-3,
    z_max=1e3,
):

    x_full = np.array(x_full0, float)

    z0 = np.exp(x_full)
    z0 = np.clip(z0, z_min, z_max)
    x_full = np.log(z0)

    n_free = len(free_idx)

    # Initial evaluation
    E, g_full = energy_and_grad(x_full, n_s, n_p)
    g = g_full[free_idx]

    # Initial approximate Hessian (identity in free subspace)
    H = np.eye(n_free)

    print(f"[BFGS] iter  0: E = {E:.10f}, ||g|| = {np.linalg.norm(g):.3e}")

    for it in range(1, max_iter + 1):
        gnorm = np.linalg.norm(g)
        if gnorm < gtol:
            print("[BFGS] Converged on gradient norm.")
            break

        # Search direction: solve H p = -g
        p = -H @ g
        if not np.all(np.isfinite(p)) or float(np.dot(g, p)) >= 0.0:
            print("[BFGS] Non-descent direction, falling back to steepest descent.")
            p = -g

        step = step0
        E_current = E
        g_dot_p = float(np.dot(g, p))

        while step > 1e-6:
            # Trial point in log-space
            x_trial = x_full.copy()
            for i, idx in enumerate(free_idx):
                x_trial[idx] += step * p[i]

            # Project onto bounds in zeta-space
            z_trial = np.exp(x_trial)
            z_trial = np.clip(z_trial, z_min, z_max)
            x_trial = np.log(z_trial)

            try:
                E_trial, g_trial_full = energy_and_grad(x_trial, n_s, n_p)
            except LA.LinAlgError:
                # Bad basis (e.g. S not positive definite) -> reduce step
                step *= 0.5
                continue

            # Armijo-like condition: sufficient decrease
            if E_trial < E_current + c1 * step * g_dot_p:
                break  # accept this trial
            step *= 0.5

        if step <= 1e-6:
            print("[BFGS] Step too small in line search, stopping.")
            break

        # Accepted step
        x_new = x_trial
        E_new = E_trial
        g_new_full = g_trial_full
        g_new = g_new_full[free_idx]

        print(
            f"[BFGS] iter {it:2d}: "
            f"E = {E_new:.10f}, dE = {E_new - E:.3e}, "
            f"||g|| = {np.linalg.norm(g_new):.3e}, step = {step:.3e}"
        )

        # BFGS update in free subspace
        s = np.array([x_new[idx] - x_full[idx] for idx in free_idx])
        y = g_new - g
        ys = float(np.dot(y, s))

        if ys > 1e-10:
            Hy = H @ y
            H = (
                H
                + ((ys + np.dot(y, Hy)) / (ys ** 2)) * np.outer(s, s)
                - (np.outer(Hy, s) + np.outer(s, Hy)) / ys
            )
        else:
            print("[BFGS] Skipping Hessian update (y·s too small or non-positive).")

        # Move to new point
        x_full = x_new
        E = E_new
        g = g_new

    return x_full, E



if __name__ == "__main__":

    Z = 4
    N_elec = 4

    zeta_s_init = [93.7103745724026, 36.31530927584123, 17.326076791934174, 9.205020166191924, 5.220119627077525, 3.0652717789292354, 1.3075479390829232, 0.708271178970315]
    #zeta_p_init = [5.328868021859073, 2.504062385838211, 1.302070852874622, 0.7528925618726321]

    n_s = len(zeta_s_init) 
    #n_p = len(zeta_p_init)  
    n_p = 0

    #x_full0 = pack_params(zeta_s_init, zeta_p_init)
    x_full0 = pack_params(zeta_s_init, [])


  
    free_idx = [i for i in range(len(x_full0))] 


    print("Free parameter indices (log ζ):", free_idx)
    x_full_opt, E_opt = quasi_newton_bfgs(
        x_full0,
        n_s,
        n_p,
        free_idx,
        max_iter=50,
        gtol=1e-5,
        step0=1.0,
        c1=1e-4,
    )

    #zeta_s_opt, zeta_p_opt = unpack_params(x_full_opt, n_s, n_p)
    zeta_s_opt = unpack_params(x_full_opt, n_s, n_p)
    zeta_s_opt, zeta_p_opt = unpack_params(x_full_opt, n_s, 0)
    zeta_s_opt = np.asarray(zeta_s_opt, dtype=float).ravel().tolist()

    print("\nOptimized zeta_s:", zeta_s_opt)
    print("Optimized zeta_p:", zeta_p_opt)

    #basis_opt = make_basis(zeta_s_opt, zeta_p_opt)
    basis_opt = make_basis(zeta_s_opt)

    result_opt = scf(basis_opt, Z, N_elec, max_iter=150, conv=1e-7, damping=0.3)

    print("\nOptimized SCF summary:")
    print(f"  Total energy        = {result_opt['E_total']:.10f}")
    print(f"  One-electron energy = {result_opt['E_one']:.10f}")
    print(f"  Two-electron energy = {result_opt['E_two']:.10f}")
    print(f"  Iterations          = {result_opt['iterations']}")
    print("  Orbital energies:")
    print(np.sort(result_opt["orb_energies"]))
'''