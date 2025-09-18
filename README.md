## NDR-Helium

A simple Python implementation of atomic self-consistent field (SCF) / Hartree-Fock calcualtion. The project currenctly demonstrates **optimization of an even-tempered Slater-type orbital (STO) basis set** for the helium ground state.

## Requirement

- Python 3.10+
- Numpy, Scipy, SymPy

## Usage
To run Run the helium Hartree–Fock SCF calculation with even-tempered optimization:

python He_optimize.py

To run the Hartree-Fock SCF calculation for Broron with enforced spherical symmetry:

python Boron.py

## Roadmap

- [x] Helium atom SCF with even-tempered basis optimization  
- [x] Generalize to s-only orbitals for arbitrary atoms  
- [ ] Extend to p-orbitals under spherical symmetry (e.g., Boron 2p averaged as 1/3 occupation)   
