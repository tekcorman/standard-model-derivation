#!/usr/bin/env python3
"""
predictions/V_count.py — |V| = 4, the vertex count of the srs lattice's
primitive cell (equivalently, the vertex count of the K_4 quotient graph).

This is one of the framework's structural primitives, alongside k_star=3,
d_spatial=3, p_toggle=2, g_girth=10. It is the proven value of |V| for
the unique MDL-optimal 3-regular 3D crystal net (the srs lattice, Sunada
2012). |V|=4 is a corollary of three theorems:

  1. MDL + Gleason force d=3 (predictions/d_spatial.py).
  2. MDL + NB-walk efficiency force k*=3 (predictions/k_star.py).
  3. Sunada 2012 Theorem 3.1: among all 3-regular 3D periodic
     crystal nets, srs is the unique strongly-isotropic minimizer —
     and srs's primitive cell has exactly 4 atoms.

Equivalently, the K_4 quotient graph (which compresses srs's directed-edge
algebra to a 4-vertex / 6-edge object encoding the C₃ generation symmetry)
has |V_K_4| = 4 vertices.

The Cl(6) Fock space's per-generation slot, the Pati-Salam SU(4) color
extension, the three generations + sterile-mode slot at the P point, the
Hashimoto sector decomposition's 4+2+2 multiplicity structure, and the
srs primitive-cell atom count all reflect this same |V|=4.

Companion-leaf: predictions/E_count.py (= 6, derivable from |V| + k*
via the handshake lemma) and predictions/g_girth.py (= 10).
"""

# ============================================================
# PARAMETER: |V| (vertex count of srs primitive cell / K_4 quotient)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Not a directly observed quantity. The framework's prediction is a
# theorem-grade structural integer (Sunada 2012 + MDL + Gleason chain).
# Cross-checks via observable consequences: 3 generations + sterile slot
# (W45 closure on m_ν₁ = 0), the C₃ at P-point fiber, etc.

# --- PREDICTED VALUE -----------------------------------------
# Value:       |V| = 4 (proven for srs case k* = 3, d = 3)
# Deviation:   N/A (structural integer; not a quantitative observable)

# --- DERIVED FORMULA -----------------------------------------
# Theorem chain:
#   Step 1 [Type 1 axiom]:    A1 (binary toggle) + P1' (observer ↔ register).
#   Step 2 [Type 3, Gleason]: minimum non-contextual Hilbert dim = 3
#                              ⇒ d_spatial = 3 (predictions/d_spatial.py).
#   Step 3 [Type 2 algebra]:  MDL + NB-walk efficiency ⇒ k* = 3
#                              (predictions/k_star.py).
#   Step 4 [Type 3, Sunada 2012]: among all 3-regular periodic
#                              d=3 crystal nets, srs is the unique
#                              strongly-isotropic minimizer.
#   Step 5 [Type 3, RCSR/srs]: the srs primitive cell has 4 atoms;
#                              equivalently, the K_4 quotient has 4
#                              vertices.
#
# The function below takes k* and d as parameters and returns 4 only
# for the proven k*=3, d=3 case (raises on any other input — the
# framework's claim of |V|=4 is conditional on the srs identification).

# --- INPUTS --------------------------------------------------
# symbol  | value | status    | predictions/ file        | meaning
# --------|-------|-----------|--------------------------|--------
# k_star  | 3     | [derived] | predictions/k_star.py    | coordination number
# d       | 3     | [derived] | predictions/d_spatial.py | spatial dimension

# --- IMPLEMENTATION ------------------------------------------

import functools


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_V_count(k_star, d):
    """
    Return the vertex count |V| of the primitive cell of the MDL-optimal
    k-regular d-dimensional crystal net.

    For k_star=3, d=3: the unique MDL-minimum crystal net is srs
    (Sunada 2012 Theorem 3.1), which has 4 atoms per primitive cell.
    Equivalently, the K_4 quotient has 4 vertices.

    Parameters
    ----------
    k_star : int
        Coordination number (from predict_k_star).
    d : int
        Spatial dimension (from predict_d_spatial).

    Returns
    -------
    int
        |V| = vertex count of the srs primitive cell / K_4 quotient.

    Raises
    ------
    ValueError
        If k_star != 3 or d != 3 (only this case is proven; the
        framework's claim of |V|=4 is conditional on srs identification).
    """
    if k_star != 3 or d != 3:
        raise ValueError(
            f"|V| = 4 only proven for k*=3, d=3 (srs). Got k={k_star}, d={d}."
        )
    # srs primitive cell: 4 atoms (Sunada 2012; RCSR-srs database).
    # K_4 quotient: 4 vertices (corollary of srs's quotient-graph
    # decomposition under the Pati-Salam embedding).
    return 4


# --- INTROSPECTION (for run_predictions.py) ------------------
# Module-scope lift so the SECTORS runner can introspect.
from k_star import predict_k_star
from d_spatial import predict_d_spatial

_d = predict_d_spatial()
_k = predict_k_star(_d)
V_count_pred = predict_V_count(_k, _d)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  |V|  --  vertex count of srs primitive cell / K_4 quotient")
    print("=" * 68)
    print(f"  k_star      = {_k}  (predictions/k_star.py)")
    print(f"  d           = {_d}  (predictions/d_spatial.py)")
    print(f"  |V|         = {V_count_pred}")
    print()
    print("  Status: framework structural integer — proven for the srs")
    print("    primitive cell (Sunada 2012 + MDL + Gleason chain).")
    print("    Companion leaves: predict_E_count (= 6), predict_g_girth (= 10).")
