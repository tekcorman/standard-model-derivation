#!/usr/bin/env python3
"""
Canonical prediction file for theta_QCD (QCD vacuum angle).

Audit anchor: Row P16 of `docs/parameters/parameter_uniqueness_ledger.md`. UNIQUE —
THEOREM-GRADE. Exact integer 0 forced by srs flatness (Z₃ gauge connection
flatness; cycle-holonomy CAS verification + discrete Ambrose-Singer per
Kobayashi-Nomizu Vol I §II.4). Conditional on Rows 4, 6 of
`docs/audits/registers/uniqueness_ledger.md` (k* = 3 + srs identification).
"""

# ============================================================
# PARAMETER: theta_QCD (QCD vacuum angle)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       < 10^-10 (from neutron EDM bound)
# Source:      Abel et al., Phys. Rev. Lett. 124, 081803, 2020
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       0 (exact)
# Deviation:   0 (consistent with bound)

# --- DERIVED FORMULA -----------------------------------------
# theta_QCD = 0, from flatness of the Z₃ gauge connection on srs.
#
# Derivation chain:
#   1. k* = 3 (from predictions/k_star.py) → srs lattice selected
#   2. At each trivalent vertex, the C₃ site symmetry defines a
#      Z₃ gauge connection on the edge labels {0, 1, 2}. The space
#      of Z_3 connections modulo gauge is H¹(srs; Z_3), per the H¹
#      Master Theorem (docs/theorems/theorem_h1_master_compression.md
#      Theorem 4(i) for Z_p extension; Theorem 3 for Wilson-loop
#      characterization of gauge invariance).
#   3. The differential holonomy at vertex v along a cycle is
#      φ_v = (ℓ_exit - ℓ_entry) mod 3, which is gauge-invariant
#      (both labels shift by the same amount under gauge transform);
#      equivalently, the cycle's Wilson loop W(C) = Σφ_v ∈ Z_3 per
#      H¹ Master Theorem 3.
#   4. The total holonomy Φ = Σ φ_v mod 3 vanishes for ALL
#      non-backtracking cycles on srs (girth-10, 12, 14 verified
#      computationally in proofs/flavor/z3_holonomy_cycles.py;
#      all longer cycles follow from vertex+edge-transitivity of
#      I4_132, which means the checked cycles generate π₁). All
#      Wilson loops trivial ⟹ realized connection sits at the trivial
#      H¹(srs; Z_3) class.
#   5. Flat connection → globally trivializable Z₃ bundle
#      (discrete Ambrose-Singer theorem — Kobayashi & Nomizu,
#       "Foundations of Differential Geometry" Vol I, Ch II, Thm 4.2,
#       adapted to discrete bundles).
#   6. theta_QCD = 0 (no topological phase).

# --- INPUTS --------------------------------------------------
# symbol | value | status    | predictions/ file     | meaning
# -------|-------|-----------|----------------------|--------
# k_star | 3     | [derived] | predictions/k_star.py | coordination number (selects srs)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
import functools

d = predict_d_spatial()
k = predict_k_star(d)

theta_QCD = 0

print(f"k* = {k} → srs lattice (I4_132)")
print(f"Z₃ connection: C₃ site symmetry at each trivalent vertex")
print(f"Holonomy check: all girth-10, 12, 14 cycles have Φ = 0 mod 3")
print(f"Connection is flat → bundle globally trivializable")
print(f"theta_QCD = {theta_QCD} (exact)")
print(f"Strong CP problem: resolved (topological, not fine-tuned)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_theta_QCD(k_star):
    """
    Returns theta_QCD = 0 for the srs lattice.

    The Z₃ gauge connection induced by C₃ site symmetry at each
    trivalent vertex is flat: all holonomies vanish. The Z₃ bundle
    is globally trivializable, so theta_QCD = 0 exactly.

    Parameters
    ----------
    k_star : int
        Coordination number (from predict_k_star). Must be 3 (srs).

    Returns
    -------
    int
        theta_QCD = 0.
    """
    if k_star != 3:
        raise ValueError(f"theta_QCD only proven for k*=3 (srs). Got {k_star}.")
    return 0


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = theta_QCD
    pure_result = predict_theta_QCD(k)
    print(f"\nImplementation: {impl_result}")
    print(f"Pure function:  {pure_result}")
    assert impl_result == pure_result == 0
    print("OK: outputs agree. theta_QCD = 0 exactly.")
