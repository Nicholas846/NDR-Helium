import numpy as np
from src import (
    make_basis, scf_loop, even_temper
)

if __name__ == "__main__":
    Z = 2
    MAX_ITER = 50
    CONV = 1e-6

    n_s = 5
    n_p = 5
    zeta_s = even_temper(n_s, 0.5, 1.6)
    zeta_p = even_temper(n_p, 0.4, 1.3)

    print(f"Using He s exponents: {zeta_s}")
    print(f"Using He p exponents: {zeta_p}")



    basis_functions = []
    basis_functions.extend(make_basis(zeta_s, [0]))  # s-shells
    basis_functions.extend(make_basis(zeta_p, [1]))  # p-shells

    occ_s = [1]
    occ_p = [1/3, 1/3, 1/3] 

    E_tot, E_one, E_two = scf_loop(basis_functions, Z, occ_s, occ_p, max_iter=MAX_ITER, conv=CONV)
    print(f"Final SCF Energy: {E_tot:.12f}  E_one = {E_one:.12f}  E_two = {E_two:.12f}")