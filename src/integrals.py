
import numpy as np
from scipy.special import gamma, gammainc, gammaincc
from scipy.integrate import quad
from sympy.physics.wigner import wigner_3j

def kinetic_integral(bi, bj):
  if bi.l != bj.l or bi.m != bj.m:
    return 0
  l = bi.l
  zeta_i, zeta_j = bi.zeta, bj.zeta

  I_1 = ((zeta_i*zeta_j) * gamma(2*l + 3))/ (zeta_i + zeta_j)**(2*l+3)
  I_2 = (l*(zeta_i +zeta_j) * gamma(2*l + 2))/ (zeta_i + zeta_j)**(2*l+2)
  I_3 = (l**2*(zeta_i +zeta_j) * gamma(2*l + 1))/ (zeta_i + zeta_j)**(2*l+1)
  I_4 = (l*(l-1)*gamma(2*l + 1)) / (zeta_i + zeta_j)**(2*l+1)

  return 0.5 * (I_1 - I_2 + I_3 - I_4)

def potential_integral(bi, bj, Z):
  if bi.l != bj.l or bi.m != bj.m:
    return 0
  l = bi.l
  zeta_i, zeta_j = bi.zeta, bj.zeta
  return - Z * gamma(2*l + 2) / (zeta_i + zeta_j)**(2*l+2)

def overlap_integral(bi, bj):
  if bi.l != bj.l or bi.m != bj.m:
    return 0
  l = bi.l
  zeta_i, zeta_j = bi.zeta, bj.zeta
  return gamma(2*l + 3) / (zeta_i + zeta_j)**(2*l+3)


def repulsion_radial(bi, bj, bz, bl, L):
    a, b = bi.l + bj.l + 2, bz.l + bl.l + 2
    p, q = bi.zeta + bj.zeta, bz.zeta + bl.zeta

    def integrant_1(r):
        return (1 / (r**(L+1) * p**(a + L + 1))) * gamma(a + L + 1) * gammainc(a + L + 1, p*r) * np.exp(-q * r) * r**b

    def integrant_2(r):
        return (r**L / p**(a - L)) * gamma(a - L) * gammaincc(a - L, p*r) * np.exp(-q * r) * r**b

    l_min = abs(bi.l - bj.l)
    l_max = bi.l + bj.l

    radial = 0

    val_1, err_1 = quad(integrant_1, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    val_2, err_2 = quad(integrant_2, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    val = val_1 + val_2
    radial += val

    return radial

def repulsion_angular(bi, bj, bz, bl, L):
    norm = np.sqrt((2*bi.l + 1) * (2*bj.l + 1) * (2*bz.l + 1) * (2*bl.l + 1))
    angular = 0

    for m in range(-L, L + 1):
        wig_1 = wigner_3j(bi.l, L, bj.l, 0, 0, 0)
        wig_2 = wigner_3j(bi.l, L, bj.l, bi.m, -m, bj.m)
        wig_3 = wigner_3j(bz.l, L, bl.l, 0, 0, 0)
        wig_4 = wigner_3j(bz.l, L, bl.l, bz.m, m, bl.m)

        if (wig_1 == 0) or (wig_2 == 0) or (wig_3 == 0) or (wig_4 == 0):
            return 0

        angular += norm * wig_1 * wig_2 * wig_3 * wig_4

    return angular

def two_electron_repulsion(bi, bj, bk, bl, L):
    r = repulsion_radial(bi, bj, bk, bl, L)
    a = repulsion_angular(bi, bj, bk, bl, L)
    return r * a