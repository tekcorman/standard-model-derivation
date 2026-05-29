#!/usr/bin/env python3
"""
Shared helper for quark Koide-ratio derivations. Not SF-compliant (no
`derives:` frontmatter, no sentinel). Imported by m_u/m_c/m_d/m_s
derivation scripts. Leading underscore marks it as a helper, not a
primary derivation script (same convention as _mssm_rge.py).

Koide parametrization per sector n in {0 (leptons), 1 (down), 2 (up)}:

    sqrt(m_j) = M0 * (1 + eps_n * cos(2*pi*j/k* + delta_n)),  j=0..k*-1

with
    eps_n^2  = 2 + 6 * alpha_1_full * n * f(n)
    delta_n  = 2 / (9 * (n + 1))
    f(n)     = 1 + (n - 1) * (g - 2) / (2 * g)
    g        = 10  (srs girth)
    k*       = 3   (srs vertex valence)

Here alpha_1_full = alpha_1_bare * tan^2(arg h) with h the Hashimoto
walker eigenvalue and tan^2(arg h) = (Im h / Re h)^2 = 5/3 at the
P-point. This matches the chirality-class factor in the lepton y_tau
derivation.

Ratio propagation: anchor the heaviest mass (factor f_max in the
ascending-sorted triplet) and compute the lighter two as

    m_mid = m_anchor * (f_mid / f_max)^2
    m_min = m_anchor * (f_min / f_max)^2

Stage 1 scope (2026-04-15): script conversion only. The formula
eps_n^2 = 2 + 6*alpha_1_full*n*f(n) and the sector anchor choices
(heaviest -> f_max) both carry conjecture-grade steps from
quark_koide_proof.py section 5-6. Grade upgrade deferred to Stage 2
(Koide delta(n) first-principles derivation; theory_open_items).
"""

import math

G_GIRTH = 10
K_STAR_DEFAULT = 3


def f_of_n(n: int, g: int = G_GIRTH) -> float:
    return 1.0 + (n - 1) * (g - 2) / (2.0 * g)


def eps_sq_of_n(n: int, alpha_1_full: float, g: int = G_GIRTH) -> float:
    return 2.0 + 6.0 * alpha_1_full * n * f_of_n(n, g)


def delta_of_n(n: int) -> float:
    return 2.0 / (9.0 * (n + 1))


def alpha_1_full_from_inputs(alpha_1_bare: float, h_real: float, h_imag: float) -> float:
    if alpha_1_bare <= 0:
        raise ValueError(f"alpha_1_bare must be > 0; got {alpha_1_bare}")
    if h_real == 0:
        raise ValueError("Re(h) must be non-zero")
    tan_sq = (h_imag / h_real) ** 2
    return alpha_1_bare * tan_sq


def koide_factors(n: int,
                  alpha_1_full: float,
                  k_star: float = K_STAR_DEFAULT,
                  g: int = G_GIRTH) -> dict:
    """Return the 3 Koide factors (ascending) plus the per-sector Koide
    parameters used to build them. f_min, f_mid, f_max correspond to
    the lightest, middle, heaviest generation in sector n."""
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")
    k_int = int(round(k_star))
    eps_sq = eps_sq_of_n(n, alpha_1_full, g)
    if eps_sq <= 0:
        raise ValueError(f"eps^2 must be > 0; got {eps_sq}")
    eps = math.sqrt(eps_sq)
    delta = delta_of_n(n)
    factors = [1.0 + eps * math.cos(2.0 * math.pi * j / k_int + delta)
               for j in range(k_int)]
    factors_sorted = sorted(factors)
    return {
        'eps_sq': eps_sq,
        'eps': eps,
        'delta': delta,
        'f_n': f_of_n(n, g),
        'factors_sorted': factors_sorted,
        'f_min': factors_sorted[0],
        'f_mid': factors_sorted[1],
        'f_max': factors_sorted[-1],
    }


def shared_inputs() -> dict:
    """Framework constants + PDG-observed anchors for the quark-Koide scripts.

    Inlined here (these scripts live under proofs/, so explicit constants are
    fine — the rigorous DAG under predictions/ derives all of these from first
    principles and is never given a hardcoded value). Formerly read from
    data/framework_inputs.yaml, removed 2026-05-28.

    The three framework constants are written in their exact algebraic form:
        alpha_1_bare = (2/3)^8                 bare NB-walk survival on srs (k*=3, g=10)
        h            = (sqrt 3 + i sqrt 5)/2   Hashimoto walker eigenvalue at P
        k_star       = 3                       srs coordination number

    m_t and m_b are PDG-2024 **observed** values used as Stage-1 anchors in
    these old-chain quark-Koide scripts. They are NOT framework predictions —
    the framework's live m_t / m_b come from the M_persistence chain in
    predictions/m_t.py and predictions/m_b.py.
    """
    return {
        'alpha_1_bare': (2.0 / 3.0) ** 8,
        'h_real': math.sqrt(3.0) / 2.0,
        'h_imag': math.sqrt(5.0) / 2.0,
        'k_star': 3.0,
        'm_t': 172.69,   # PDG 2024 observed (pole)        — anchor for m_u, m_c
        'm_b': 4.18,     # PDG 2024 observed (MS-bar @ m_b) — anchor for m_d, m_s
    }
