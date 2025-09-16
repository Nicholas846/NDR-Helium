import numpy as np
from .basis import make_basis
from .matrices import S_matrix, H_matrix, ERI_tensor
from .scf import scf_loop

def even_temper(n, alpha0, beta):
    return alpha0 * beta ** np.arange(n, dtype=float)

def _energy_given_zetas(zetas, Z, occ):
    bfs = make_basis(zetas, [0])  # s-only for minimal example
    S = S_matrix(bfs)
    H = H_matrix(bfs, Z)
    ERI = ERI_tensor(bfs)
    E_tot, E_one, E_two = scf_loop(bfs, Z, occ, S, H, ERI, max_iter=200, conv=1e-9)
    return E_tot

def optimize_zeta_minimal(n, Z, occ,
                          u0=np.log(0.5), v0=np.log(1.5 - 1.0),
                          ha=1e-3, hb=1e-3, tol_E=1e-8, tol_step=1e-6, max_iter=15):
    # alpha0 = exp(u) > 0, beta = 1 + exp(v) > 1
    def energy_from_uv(u, v):
        a = np.exp(u)
        b = 1.0 + np.exp(v)
        zetas = even_temper(n, a, b)
        return float(_energy_given_zetas(zetas, Z, occ))

    def fd(E, u, v, hu, hv):
        E0   = E(u, v)
        Eu_p = E(u+hu, v); Eu_m = E(u-hu, v)
        Ev_p = E(u, v+hv); Ev_m = E(u, v-hv)
        g_u  = (Eu_p - Eu_m)/(2*hu)
        g_v  = (Ev_p - Ev_m)/(2*hv)
        H_uu = (Eu_p - 2*E0 + Eu_m)/(hu**2)
        H_vv = (Ev_p - 2*E0 + Ev_m)/(hv**2)
        Epp  = E(u+hu, v+hv); Epm = E(u+hu, v-hv)
        Emp  = E(u-hu, v+hv); Emm = E(u-hu, v-hv)
        H_uv = (Epp - Epm - Emp + Emm)/(4*hu*hv)
        return E0, np.array([g_u, g_v]), np.array([[H_uu, H_uv], [H_uv, H_vv]])

    u, v = float(u0), float(v0)
    E, g, H = fd(energy_from_uv, u, v, ha, hb)
    for _ in range(max_iter):
        step = -np.linalg.solve(H + 1e-8*np.eye(2), g)
        u_new, v_new = u + step[0], v + step[1]
        E_new, g_new, H_new = fd(energy_from_uv, u_new, v_new, ha, hb)
        if abs(E - E_new) < tol_E or np.linalg.norm(step) < tol_step:
            u, v, E, g, H = u_new, v_new, E_new, g_new, H_new
            break
        u, v, E, g, H = u_new, v_new, E_new, g_new, H_new

    alpha0 = np.exp(u)
    beta   = 1.0 + np.exp(v)
    return {"alpha0": alpha0, "beta": beta, "energy": E, "zetas": even_temper(n, alpha0, beta)}
