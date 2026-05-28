import numpy as np
from src import make_basis
from src.solver.hf_scf import scf
from src.solver.ci_full import FullCISolver
from src.solver.orbital_opt import optimize_hf_orbitals
from src.solver.NDR import calculate_1rdm, get_natural_orbitals, calculate_2rdm, partial_trace_2rdm

def run_calculation(Z, N_elec, zetas, mode="hf"):
    """
    mode: "hf"          -> Single Hartree-Fock run
          "opt"         -> Optimize zetas for HF energy
          "fci"         -> Run HF then Full CI
    """
    print(f"\n{'='*60}")
    print(f" SYSTEM: Z={Z}, N={N_elec} | MODE: {mode.upper()}")
    print(f"{'='*60}")

    if mode == "opt":
        print("--> Starting Orbital Optimization...")
        res = optimize_hf_orbitals(Z, N_elec, zetas)
        working_zetas = res['zetas']
        print(f"--> Optimization Converged. Final Energy: {res['energy']:.10f}")

        print("Optimized Zetas:")
        for i, zeta in enumerate(working_zetas[0]):
            print(f"  Zeta {i+1}: {zeta:.6f}")
    else:
        working_zetas = zetas

    # 2. RUN BASE HF 
    print("--> Running SCF...")
    basis = make_basis(*working_zetas)
    scf_res = scf(basis, Z, N_elec)
    
    print(f"HF Total Energy: {scf_res['E_total']:.10f} Ha")

    # 3. FCI & NDR MODE
    if mode in ["fci", "ndr"]:
        print("--> Running Full Configuration Interaction...")
        ci_engine = FullCISolver(scf_res)
        ci_res = ci_engine.solve()
        
        print(f"FCI Total Energy: {ci_res['E_ci']:.10f} Ha")
        print(f"Correlation Energy: {ci_res['E_correlation']:.10f} Ha")

        ground_state_vector = ci_res['vectors'][:, 0]
        print(f"Leading CI Coefficients: {ground_state_vector[0]:.4f}, {ground_state_vector[1]:.4f}, ...")

        print("\n--> Verifying Eigenvector")
        H_ci = ci_res['H_ci']
        E_ci = ci_res['E_ci']

        residual = np.dot(H_ci, ground_state_vector) - E_ci * ground_state_vector
        residual_norm = np.linalg.norm(residual)
        print(f"  FCI Hamiltonian Residual Norm (||HC - EC||): {residual_norm:.12e}")
        
        if residual_norm < 1e-10:
            print("  [SUCCESS] Eigenvector is mathematically exact for this configuration space.")
        else:
            print("  [WARNING] High residual detected. Check alignment of your operators.")
        print(f"\nCISD Components extracted from FCI:")
        
        coeffs = ci_res['cisd_coeffs']
        
        print(f"  REFERENCE (C0): {coeffs['C_0']:.6f}")
        
        for level in ['singles', 'doubles']:
            data = coeffs.get(level, [])
            if data:
                print(f"  {level.upper()}:")
                
                sorted_excitations = sorted(data, key=lambda x: abs(x[1]), reverse=True)
                for det, coeff in sorted_excitations[:5]: 
                    print(f"    Det {bin(det)}: {coeff:.6f}")

        if mode == "ndr":
            print("\n--> Analyzing Natural Determinant Reference (NDR)...")
            ground_state_vec = ci_res['vectors'][:, 0]
            
            # Compute the exact 1RDM from CI coefficients
            rdm = calculate_1rdm(ci_engine, ground_state_vec)
            print(f"1RDM Computed. Trace (electrons): {np.trace(rdm):.4f}")

            rdm2 = calculate_2rdm(ci_engine, ground_state_vec)

            pair_trace = 0.0
            for p in range(ci_engine.n_spin):
                for q in range(ci_engine.n_spin):
                    pair_trace += rdm2[p, q, p, q]
            expected_pairs = N_elec * (N_elec - 1)
            print(f"2RDM Pair Trace: {pair_trace:.4f}")

            rdm1_reconstructed = partial_trace_2rdm(ci_engine, rdm2)
            rdm_norm = rdm * 0.5
            
            # Check if the partial trace perfectly matches your native 1-RDM
            # rdm was the 1-RDM computed directly from the CI vector earlier
            if np.allclose(rdm_norm, rdm1_reconstructed, atol=1e-8):
                print("  [SUCCESS] Partial trace of 2-RDM perfectly reproduces the 1-RDM!")
                print(f"            Reconstructed 1-RDM Trace: {np.trace(rdm1_reconstructed):.4f}")
            else:
                print("  [FAIL] 1-RDM reconstruction mismatch. Check your index tracking order.")

            h_spin = ci_engine.h_spin   # One-body core Hamiltonian matrix
            g_spin = ci_engine.g_spin   # Two-body ERI tensor in Dirac notation
            
            # 1. Contract 1-RDM with one-body integrals
            # rdm1_reconstructed is your normalized 1-RDM (Trace = 1)
            one_body_energy = np.einsum('pq,pq->', rdm1_reconstructed, h_spin)
            
            # 2. Contract 2-RDM with two-body integrals
            # rdm2 is your normalized 2-RDM (Trace = 1)
            two_body_energy = np.einsum('pqrs,pqrs->', g_spin, rdm2)
            
            # 3. Apply the scaling multipliers for Filipp's normalized matrices
            # E = N * E_1 + N*(N-1) * E_2
            N = ci_engine.n_elec
            E_from_rdms = (N * one_body_energy) + (0.5 * N * (N - 1) * two_body_energy)
            
            print(f"  Energy computed from RDMs: {E_from_rdms:.10f} Ha")
            print(f"  Target Exact FCI Energy:   {ci_res['E_ci']:.10f} Ha")
            
            energy_difference = abs(E_from_rdms - ci_res['E_ci'])
            print(f"  Absolute Energy Delta:     {energy_difference:.12e}")
            
            if energy_difference < 1e-9:
                print("  [SUCCESS] RDM-derived energy matches your exact Full CI energy!")
            else:
                print("  [FAIL] Energy mismatch. Re-verify the index sequencing of your tensors.")
            # Diagonalize 1RDM to find Natural Orbitals and Occupations
            occs, natural_orbitals, ndr_coeffs = get_natural_orbitals(rdm, N_elec)
            
            print("\nNatural Orbital Occupation Numbers (top 6):")
            for i, occ in enumerate(occs[:6]):
                print(f"  NO {i+1}: {occ:.8f}")

            # The NDR consists of the most occupied natural orbitals
            print(f"\nNDR constructed from {N_elec} most occupied natural orbitals.")
            
            ci_res['1rdm'] = rdm
            ci_res['no_occupations'] = occs
            ci_res['natural_orbitals'] = natural_orbitals

        return ci_res

    return scf_res

if __name__ == "__main__":

    Z = 2
    N = 2

    my_zetas = [
        [8.955016, 2.975601, 1.477575, 0.706409, 0.207456, 0.101581],
        [1.0],
        [0.7]
    ]

    result = run_calculation(Z, N, my_zetas, mode="ndr")
