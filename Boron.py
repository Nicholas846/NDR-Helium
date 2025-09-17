import numpy as np
from atomic_scf import (
    make_basis, S_matrix, H_matrix, ERI_tensor, scf_loop,
    even_temper, optimize_zeta_minimal
)

if __name__ == "Boron":
    
    Z = 5
    MAX_ITER = 100
    CONV = 1e-6

    n_s = 2
    n_p = 3
    occ_opt = np.zeros(n_s + n_p)
    

    zeta_s = even_temper(n_s, 0.25, 2.0)
    zeta_p = even_temper(n_p, 0.2, 1.5)

    print("Calculate ground state energy of Boron")

    bfs = []
    for z in zeta_s:
        bfs.extend(make_basis([z], [0]))   
    
    for z in zeta_p:
        bfs.extend(make_basis([z], [1]))   

nbf = len(bfs)

occ = np.zeros(nbf)
occ[:2] = 2.0
occ[2:5] = 1/3


S = S_matrix(bfs)
H = H_matrix(bfs, Z)
ERI = ERI_tensor(bfs)


E_tot, E_one, E_two = scf_loop(bfs, Z, occ, S, H, ERI, max_iter=MAX_ITER, conv=CONV)

print("\n== Boron (Z=5) results ==")
print(f"E_total = {E_tot:.12f}  (one-e = {E_one:.12f}, two-e = {E_two:.12f})")
print(f"NBasis = {nbf}  (s+p; ns={n_s}, np={n_p})")
print(f"s exponents: {zeta_s}")
print(f"p exponents: {zeta_p}")





