import ast
from src import make_basis, scf

def get_input(prompt, default):
    s = input(prompt).strip()
    return ast.literal_eval(s) if s else default

if __name__ == "__main__":
    Z = get_input("Atomic number Z (default 2): ", 2)
    N_elec = get_input("Number of electrons (default 2): ", 2)

    zeta_lists = []
    shell_names = ["s", "p", "d", "f", "g"]

    for shell in shell_names:
        zetas = get_input(
            f"Enter {shell}-type zetas (empty list [] to stop): ",
            []
        )
        if not zetas:          
            break
        zeta_lists.append(zetas)

    if not zeta_lists:
        raise ValueError("At least one shell (s) must be provided.")

    basis = make_basis(*zeta_lists)

    print("\nBasis functions:")
    for bf in basis:
        print(" ", bf)

    res = scf(basis, Z, N_elec)

    print("\n=== SCF Results ===")
    print(f"E_total = {res['E_total']:.12f}")
    print(f"E_one   = {res['E_one']:.12f}")
    print(f"E_two   = {res['E_two']:.12f}")
    print(f"iters   = {res['iterations']}")
