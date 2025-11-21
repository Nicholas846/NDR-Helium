import numpy as np
from src import (
    make_basis, scf_loop, even_temper
)

def get_input(prompt, default):
    user_input = input(prompt)
    if user_input.strip():
        return eval(user_input)
    else:
        return default
    

if __name__ == "__main__":
    Z = get_input("Enter atomic number Z (default 2 for Helium): ", 2)
    MAX_ITER = 50
    CONV = 1e-6

    # How many functions (used if choosing even temper; ignored if you give manual lists)
    n_s = get_input("Enter number of s-type basis functions (default 5): ", 5)
    n_p = get_input("Enter number of p-type basis functions (default 0): ", 0)

    # Choose method per shell (True = even temper, False = manual list)
    use_even_s = get_input("Use even-temper for s? (True/False, default True): ", True)
    use_even_p = get_input("Use even-temper for p? (True/False, default True): ", True)

    # s-type zetas
    if use_even_s:
        zeta_s = even_temper(n_s, 0.4, 1.8)
    else:
        zeta_s = get_input("Enter zeta list for s (e.g. [0.3, 0.8, 2.0]; default [0.25,0.5,1.0,2.0,4.0]): ",
                           [0.25, 0.5, 1.0, 2.0, 4.0])
        n_s = len(zeta_s)

    # p-type zetas
    if use_even_p:
        zeta_p = even_temper(n_p, 0.25, 1.8) if n_p > 0 else []
    else:
        zeta_p = get_input("Enter zeta list for p (e.g. [0.2, 0.6, 1.8]; default []): ", [])
        n_p = len(zeta_p)

    basis_functions = []
    if n_s > 0:
        basis_functions.extend(make_basis(zeta_s, [0]))  # s (l=0)
    if n_p > 0:
        basis_functions.extend(make_basis(zeta_p, [1]))  # p (l=1)

    print("Basis functions:")
    for bf in basis_functions:
        print(" ", bf)

    occ_s = get_input("Enter occupation numbers for s orbitals as a list (default [2]): ", [2])
    occ_p = get_input("Enter occupation numbers for p orbitals as a list (default []): ", [])

    E_tot, E_one, E_two, E_two_2, Norm_check = scf_loop(
        basis_functions, Z, occ_s, occ_p, max_iter=MAX_ITER, conv=CONV
    )
    print(f"Final SCF Energy: {E_tot:.12f}  E_one = {E_one:.12f}  E_two = {E_two:.12f} "
          f"E_two_2 = {E_two_2:.12f} Norm check = {Norm_check:.12f}")
