import numpy as np
from src import (
    make_basis, scf_loop, even_temper
)

if __name__ == "__main__":
    Z = 5
    MAX_ITER = 50
    CONV = 1e-6


    n_s = 10
    n_p = 10

    # even-tempered exponents
    zeta_s = even_temper(n_s, 0.25, 5)
    zeta_p = even_temper(n_p, 0.2, 2.0)

    '''zeta_s = [250.0, 62.5, 15.625, 3.90625, 0.9765625]
    zeta_p = [125.0, 31.25, 7.8125, 1.953125, 0.48828125]'''

    print("Calculate ground state energy of Boron (Z=5)")

    basis_functions = []
    basis_functions.extend(make_basis(zeta_s, [0]))  # s-shells
    basis_functions.extend(make_basis(zeta_p, [1]))  # p-shells (3 per zeta)

    print("Basis functions:")
    for bf in basis_functions:
        print(" ", bf)

    
    occ_s = [2, 2] + [0]*8
    occ_p = [1/3, 1/3, 1/3] + [0]*7

    E_tot, E_one, E_two = scf_loop(basis_functions, Z, occ_s, occ_p, max_iter=MAX_ITER, conv=CONV)
    print(f"Final SCF Energy: {E_tot:.12f}  E_one = {E_one:.12f}  E_two = {E_two:.12f}")