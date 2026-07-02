#!/usr/bin/env python3
"""
Koide factor helpers for the framework's within-species mass triplets.

Leading underscore marks this as an internal helper (not a primary
parameter prediction). Imported by predictions/m_{c,u,s,d}.py.

Per the framework's quark Koide cosine parametrization (both within-species
phase δ(n) AND amplitude ε(n) now theorem-grade-structural):

    sqrt(m_j) = sqrt(M0) * (1 + eps_n * cos(2*pi*j/k* + delta_n)),  j=0..k*-1

with
    eps_n²  = 2 + 6 * alpha_1_full * n * f(n)
             [THEOREM-GRADE-STRUCTURAL per
              ../docs/theorems/theorem_quark_koide_eps_n_2026-05-26.md (W4)]
    delta_n = 2 / (9 * (n + 1))
             [THEOREM-GRADE-STRUCTURAL per
              ../docs/theorems/theorem_W3_PS_sector_connectivity_2026-05-26.md (W3)]
    f(n)    = 1 + (n - 1) * (g - 2) / (2 * g)
    g       = 10  (srs girth, theorem-grade per ../predictions/g_girth.py)
    k*      = 3   (srs vertex valence, theorem-grade per ../predictions/k_star.py)

Within-species sector index n (theorem-grade per W3):
    n = 0   leptons (charged) — PS graph distance 0 from L
    n = 1   down-type quarks (d, s, b) — PS graph distance 1 via SU(4) leptoquark
    n = 2   up-type quarks (u, c, t) — PS graph distance 2 via SU(2)_L doublet
    n = 3   (charged lepton — same Koide structure as n=0 by C-symmetry)

The "6" coefficient in eps_n² is the PS leptoquark coset dimension:
    N_LQ = dim SU(4)/(SU(3)·U(1)) = 15 - 8 - 1 = 6
identified per W4 theorem doc with the broken SU(4)_PS generators mediating
inter-sector Koide-deviation. The structural derivation chain is:

    N_LQ (Type 2 algebra)
    × alpha_1_full per channel (Type 1 A5(b) + Type 4 alpha_1_full.py)
    × Schur's lemma gauge equivariance → equal contribution per channel (Type 3)
    × n·f(n) many-body cluster expansion with pair-corr (g-2)/g (Type 3 + Type 4)
    = 6 · alpha_1_full · n · f(n)

Mass propagation: anchor the heaviest mass (factor f_max in the ascending-
sorted triplet) and compute the lighter two via

    m_mid = m_anchor * (f_mid / f_max)^2
    m_min = m_anchor * (f_min / f_max)^2

All inputs framework-internal (alpha_1_full, k_star, g_girth), no
empirical constants except those derived via the framework chain.

GRADE HISTORY:
  2026-04-15: Stage 1 script conversion; eps_n² and anchor choices were
              flagged "conjecture-grade, grade upgrade deferred to Stage 2"
              (see proofs/masses/_quark_koide.py historical docstring).
  2026-05-26 (this update): both δ(n) [W3] and ε(n) [W4] are now
              THEOREM-GRADE-STRUCTURAL. The "Stage 2 upgrade deferred"
              admission is closed. Verification probes:
                  proofs/masses/W3_PS_sector_connectivity_2026-05-26.py (δ)
                  proofs/foundations/W27_eps_n_theorem_closure_2026-05-26.py (ε)
              Both pass 7/7 pre-declared gates.
"""

import math
import functools


@functools.lru_cache(maxsize=None)
def koide_f_of_n(n, g_girth):
    """Sector breaking prefactor f(n) = 1 + (n-1)·(g-2)/(2g)."""
    return 1.0 + (n - 1) * (g_girth - 2) / (2.0 * g_girth)


@functools.lru_cache(maxsize=None)
def koide_eps_sq(n, alpha_1_full, g_girth):
    """ε²(n) = 2 + 6·α₁_full·n·f(n).  Lepton (n=0): ε² = 2 exactly."""
    return 2.0 + 6.0 * alpha_1_full * n * koide_f_of_n(n, g_girth)


@functools.lru_cache(maxsize=None)
def koide_delta_n(n):
    """δ(n) = 2/(9(n+1)) — W3 theorem-grade-structural."""
    return 2.0 / (9.0 * (n + 1))


def koide_factors(n, alpha_1_full, k_star, g_girth):
    """
    Three Koide factors ascending: (f_min, f_mid, f_max).
    f_j = 1 + ε_n · cos(2πj/k* + δ_n) for j = 0, 1, ..., k*-1.

    Returns dict with f_min, f_mid, f_max, eps_sq, delta, f_n.
    """
    eps_sq = koide_eps_sq(n, alpha_1_full, g_girth)
    if eps_sq <= 0:
        raise ValueError(f"eps² must be > 0; got {eps_sq} for n={n}")
    eps = math.sqrt(eps_sq)
    delta = koide_delta_n(n)
    k_int = int(round(k_star))
    factors = [1.0 + eps * math.cos(2.0 * math.pi * j / k_int + delta)
               for j in range(k_int)]
    factors_sorted = sorted(factors)
    return {
        'eps_sq': eps_sq,
        'eps': eps,
        'delta': delta,
        'f_n': koide_f_of_n(n, g_girth),
        'f_min': factors_sorted[0],
        'f_mid': factors_sorted[1],
        'f_max': factors_sorted[-1],
    }


def koide_lighter_mass(m_anchor, n, position, alpha_1_full, k_star, g_girth):
    """
    Compute the lighter mass from the heaviest mass via Koide cosine ratio.

    Parameters
    ----------
    m_anchor : float       Heaviest generation mass in the sector.
    n : int                Sector index (0 lepton, 1 down, 2 up).
    position : str         'mid' or 'min' — which lighter generation.
    alpha_1_full, k_star, g_girth : framework-internal inputs.

    Returns
    -------
    float                  Predicted lighter-generation mass (same units as m_anchor).
    """
    koide = koide_factors(n, alpha_1_full, k_star, g_girth)
    if position == 'mid':
        ratio_sq = (koide['f_mid'] / koide['f_max']) ** 2
    elif position == 'min':
        ratio_sq = (koide['f_min'] / koide['f_max']) ** 2
    else:
        raise ValueError(f"position must be 'mid' or 'min'; got {position!r}")
    return m_anchor * ratio_sq


if __name__ == "__main__":
    # Sanity check: lepton (n=0), down (n=1), up (n=2) Koide structure
    alpha_1_full_val = (5.0 / 3.0) * (2.0 / 3.0) ** 8
    print(f"alpha_1_full = (5/3)(2/3)^8 = {alpha_1_full_val:.8f}")
    print()
    print(f"{'Sector':<8} {'n':>3} {'ε²':>10} {'δ rad':>10} {'δ deg':>10} {'f_max':>10} {'f_mid':>10} {'f_min':>10}")
    for sector, n in [("lepton", 0), ("down", 1), ("up", 2)]:
        k = koide_factors(n, alpha_1_full_val, 3, 10)
        print(f"{sector:<8} {n:>3} {k['eps_sq']:>10.6f} {k['delta']:>10.6f} "
              f"{math.degrees(k['delta']):>10.4f} "
              f"{k['f_max']:>10.6f} {k['f_mid']:>10.6f} {k['f_min']:>10.6f}")
