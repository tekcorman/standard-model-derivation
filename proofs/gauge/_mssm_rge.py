"""MSSM two-loop RGE helper used by the gauge-sector SF derivations.

Not an SF-compliant script on its own — it carries no `derives:` frontmatter
and emits no sentinel. Its job is to provide a single, authoritative
implementation of the two-loop MSSM/SM renormalization-group running of the
gauge couplings (alpha_1, alpha_2, alpha_3) in GUT normalization, so that
alpha_GUT_derivation.py, g_1_derivation.py, g_2_derivation.py, and
g_3_derivation.py all agree to machine precision.

Conventions
-----------
alpha_1 is in GUT normalization: alpha_1 = (5/3) * alpha_Y,
where alpha_Y = g'^2 / (4 pi) is the Standard Model hypercharge coupling.

Two-loop beta functions (from Martin & Vaughn 1993, sign conventions as in
Peskin/Schroeder):

  d(alpha_i^{-1}) / d(ln mu) = -b_i / (2 pi)
                               - sum_j b_ij * alpha_j / (8 pi^2)

MSSM one-loop (above M_SUSY):    b = [33/5, 1, -3]
SM   one-loop (below M_SUSY):    b = [41/10, -19/6, -7]

Two-loop matrices are the standard MSSM / SM forms.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


B_MSSM = np.array([33.0 / 5.0, 1.0, -3.0])

BIJ_MSSM = np.array([
    [199.0 / 25.0, 27.0 / 5.0, 88.0 / 5.0],
    [9.0 / 5.0,    25.0,       24.0      ],
    [11.0 / 5.0,   9.0,        14.0      ],
])

B_SM = np.array([41.0 / 10.0, -19.0 / 6.0, -7.0])

BIJ_SM = np.array([
    [199.0 / 50.0, 27.0 / 10.0, 44.0 / 5.0],
    [9.0 / 10.0,   35.0 / 6.0,  12.0      ],
    [11.0 / 10.0,  9.0 / 2.0,   -26.0     ],
])


def _rge(t, y, b, bij):
    alpha = 1.0 / y
    dydt = np.zeros(3)
    for i in range(3):
        dydt[i] = -b[i] / (2.0 * np.pi)
        for j in range(3):
            dydt[i] -= bij[i, j] * alpha[j] / (8.0 * np.pi ** 2)
    return dydt


def _integrate(t0, t1, y0, b, bij):
    sol = solve_ivp(
        _rge, [t0, t1], y0,
        args=(b, bij),
        method='RK45', rtol=1e-11, atol=1e-13, dense_output=True,
    )
    return sol.sol(t1)


def run_down_from_gut(alpha_gut_inv: float, m_gut: float, m_z: float,
                     m_susy: float = 1000.0) -> np.ndarray:
    """Run [alpha_1^-1, alpha_2^-1, alpha_3^-1] from M_GUT down to M_Z."""
    t_gut = np.log(m_gut)
    t_susy = np.log(m_susy)
    t_z = np.log(m_z)
    y0 = np.array([alpha_gut_inv, alpha_gut_inv, alpha_gut_inv])
    y_susy = _integrate(t_gut, t_susy, y0, B_MSSM, BIJ_MSSM)
    y_mz = _integrate(t_susy, t_z, y_susy, B_SM, BIJ_SM)
    return y_mz


def run_up_from_mz(alpha_inv_mz: np.ndarray, m_z: float, m_gut: float,
                   m_susy: float = 1000.0) -> np.ndarray:
    """Run [alpha_1^-1, alpha_2^-1, alpha_3^-1] from M_Z up to M_GUT."""
    t_z = np.log(m_z)
    t_susy = np.log(m_susy)
    t_gut = np.log(m_gut)
    y_susy = _integrate(t_z, t_susy, np.asarray(alpha_inv_mz, dtype=float),
                        B_SM, BIJ_SM)
    y_gut = _integrate(t_susy, t_gut, y_susy, B_MSSM, BIJ_MSSM)
    return y_gut


def observed_alpha_inv_at_mz(alpha_em: float, sin2_tw: float,
                             alpha_s: float) -> np.ndarray:
    """Build GUT-normalized inverse couplings from SM inputs at M_Z."""
    alpha_2 = alpha_em / sin2_tw
    alpha_Y = alpha_em / (1.0 - sin2_tw)
    alpha_1 = (5.0 / 3.0) * alpha_Y
    return np.array([1.0 / alpha_1, 1.0 / alpha_2, 1.0 / alpha_s])


def find_alpha_gut(alpha_em: float, sin2_tw: float, alpha_s: float,
                   m_z: float, m_susy: float = 1000.0) -> tuple:
    """Find (alpha_GUT, M_GUT) where alpha_1 and alpha_2 meet running up from M_Z."""
    alpha_inv_mz = observed_alpha_inv_at_mz(alpha_em, sin2_tw, alpha_s)
    t_z = np.log(m_z)
    t_susy = np.log(m_susy)
    y_susy = _integrate(t_z, t_susy, alpha_inv_mz, B_SM, BIJ_SM)

    def spread(log_mu):
        y = _integrate(t_susy, log_mu, y_susy, B_MSSM, BIJ_MSSM)
        return y[0] - y[1]

    log_gut = brentq(spread, np.log(1e14), np.log(1e18),
                     rtol=1e-12, xtol=1e-14)
    y_gut = _integrate(t_susy, log_gut, y_susy, B_MSSM, BIJ_MSSM)
    alpha_gut_inv = 0.5 * (y_gut[0] + y_gut[1])
    return 1.0 / alpha_gut_inv, float(np.exp(log_gut))


def couplings_at_mz_from_gut(alpha_gut_inv: float, m_gut: float, m_z: float,
                             m_susy: float = 1000.0) -> dict:
    """Return (alpha_i, g_i) at M_Z running DOWN from a given alpha_GUT."""
    y_mz = run_down_from_gut(alpha_gut_inv, m_gut, m_z, m_susy)
    alpha_1, alpha_2, alpha_3 = 1.0 / y_mz[0], 1.0 / y_mz[1], 1.0 / y_mz[2]
    # g_1 in SM hypercharge normalization (g') = sqrt(4 pi * (3/5) * alpha_1_GUT)
    g_1_sm = float(np.sqrt(4.0 * np.pi * (3.0 / 5.0) * alpha_1))
    g_2 = float(np.sqrt(4.0 * np.pi * alpha_2))
    g_3 = float(np.sqrt(4.0 * np.pi * alpha_3))
    return {
        'alpha_1_GUT': float(alpha_1),
        'alpha_2': float(alpha_2),
        'alpha_3': float(alpha_3),
        'g_1': g_1_sm,
        'g_2': g_2,
        'g_3': g_3,
    }
