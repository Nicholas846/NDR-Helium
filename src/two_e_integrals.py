import numpy as np
from scipy.special import gamma, gammainc, gammaincc
from scipy.integrate import quad
from sympy.physics.wigner import wigner_3j, clebsch_gordan, gaunt
import math
from .basis import BasisFunction, make_basis




def pair_key(li, zeta_i, lj, zeta_j):

   return tuple(sorted([(li, zeta_i), (lj, zeta_j)]))

def radial_key(bi, bj, bz, bl, L):
   
    pair_ij = pair_key(bi.l, bi.zeta, bj.l, bj.zeta)
    pair_zl = pair_key(bz.l, bz.zeta, bl.l, bl.zeta)
    
    pair_lo, pair_hi = sorted([pair_ij, pair_zl])

    return (pair_lo, pair_hi, L)

radial_cache = {}


def repulsion_radial(bi, bj, bz, bl, L):

    key = radial_key(bi, bj, bz, bl, L)

    if key in radial_cache:
        return radial_cache[key]
    
    a, b = bi.l + bj.l + 2, bz.l + bl.l + 2
    p, q = bi.zeta + bj.zeta, bz.zeta + bl.zeta
    def integrant_1(r):
      A = 1/(r**(L+1))
      B = 1/(p**(a + L + 1))
      C = gamma(a + L + 1) * gammainc(a + L + 1, p*r)
      D = np.exp(-q*r) * r**b
      return A * B * C * D

    def integrant_2(r):
      E = r**L
      F = 1/(p**(a - L))
      G = gamma(a - L) * gammaincc(a - L, p*r)
      H = np.exp(-q*r) * r**b
      return E * F * G * H
    
    I_1, _ = quad(integrant_1, 0, np.inf, limit=200)
    I_2, _ = quad(integrant_2, 0, np.inf, limit=200)
    val  = (I_1 + I_2)

    radial_cache[key] = val

    return val

def ERI(bi, bj, bz, bl):
    li, lj, lk, ll = bi.l, bj.l, bz.l, bl.l
    mi, mj, mk, ml = bi.m, bj.m, bz.m, bl.m

    Lmin = max(abs(li - lj), abs(lk - ll))
    Lmax = min(li + lj, lk + ll)

    total = 0.0

    for L in range(Lmin, Lmax + 1):
        g1 = wigner_3j(li, lj, L, 0, 0, 0)
        g2 = wigner_3j(lk, ll, L, 0, 0, 0)

        if g1 == 0.0 or g2 == 0.0:
            continue
        
        R_L = repulsion_radial(bi, bj, bz, bl, L)

        A_L = 0.0


        M = mi - mj
        if (mi + ml) == (mj + mk):
            g3 = wigner_3j(li, lj, L, mi, -mj, -M)
            g4 = wigner_3j(lk, ll, L, mk, -ml, M)
            A_L = g3 * g4 * (-1)**M
        
        if A_L != 0.0:
            total += R_L * A_L * g1 * g2
    
    pre = (-1)**(mi + mj) * np.sqrt((2*li + 1)*(2*lj + 1)*(2*lk + 1)*(2*ll + 1))
    total = pre * total

    return float(total)
