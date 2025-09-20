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

    n_s = get_input("Enter number of s-type basis functions (default 5): ", 5)
    n_p = get_input("Enter number of p-type basis functions (default 0): ", 0)

    zeta_s = even_temper(n_s, 0.25, 5)
    zeta_p = even_temper(n_p, 0.2, 2.0)

    basis_functions = []
    basis_functions.extend(make_basis(zeta_s, [0]))  
    basis_functions.extend(make_basis(zeta_p, [1])) 

    print("Basis functions:")
    for bf in basis_functions:
        print(" ", bf)

    
    occ_s = get_input("Enter occupation numbers for s orbitals as a list (default [2]): ", [2])
    occ_p = get_input("Enter occupation numbers for p orbitals as a list (default []): ", [])

    E_tot, E_one, E_two = scf_loop(basis_functions, Z, occ_s, occ_p, max_iter=MAX_ITER, conv=CONV)
    print(f"Final SCF Energy: {E_tot:.12f}  E_one = {E_one:.12f}  E_two = {E_two:.12f}")