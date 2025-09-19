import numpy as np
from src import (
    make_basis, scf_loop, even_temper
)

if __name__ == "__main__":
    Z = 2
    MAX_ITER = 50
    CONV = 1e-6

    n_s = 8
    zeta_s = even_temper(n_s, 0.5, 1.6)

    print(f"Using He s exponents: {zeta_s}")

    basis_functions = []
    basis_functions.extend(make_basis(zeta_s, [0]))  # s-shells

    print("Basis functions:")
    for bf in basis_functions:
        print(" ", bf)

    occ_s = [1,1]
    occ_p = [] 

    E_tot, E_one, E_two = scf_loop(basis_functions, Z, occ_s, occ_p, max_iter=MAX_ITER, conv=CONV)
    print(f"Final SCF Energy: {E_tot:.12f}  E_one = {E_one:.12f}  E_two = {E_two:.12f}")