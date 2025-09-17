# main.py — Helium (He) with s-only exponent optimization + SCF
import numpy as np
from atomic_scf import (
    make_basis, S_matrix, H_matrix, ERI_tensor, scf_loop,
    even_temper, optimize_zeta_minimal
)

if __name__ == "__He_optimize__":
    # === System ===
    Z = 2                  # Helium
    MAX_ITER = 100
    CONV = 1e-6

    # === Optimize s-only even-tempered exponents ===
    n_s = 10                                
    occ_opt = np.zeros(n_s); occ_opt[0] = 2.0  

    print("Optimizing He s-exponents (even-tempered)...")
    opt = optimize_zeta_minimal(n=n_s, Z=Z, occ=occ_opt)
    zeta_s = opt["zetas"]
    print(f"Optimized He s zetas: {zeta_s}")

    # === Build final s-only basis with optimized exponents ===
    bfs = []
    for z in zeta_s:
        bfs.extend(make_basis([z], [0]))   # l = 0 only

    nbf = len(bfs)

    # Occupations: 2 electrons → 1 doubly occupied spatial MO
    occ = np.zeros(nbf)
    occ[0] = 2.0

    # === Matrices & SCF ===
    S = S_matrix(bfs)
    H = H_matrix(bfs, Z)
    ERI = ERI_tensor(bfs)

    E_tot, E_one, E_two = scf_loop(
        bfs, Z, occ, S, H, ERI, max_iter=MAX_ITER, conv=CONV
    )

    print("\n== He results ==")
    print(f"E_total = {E_tot:.12f}  (one-e = {E_one:.12f}, two-e = {E_two:.12f})")
    print(f"NBasis = {nbf} (s only)")
