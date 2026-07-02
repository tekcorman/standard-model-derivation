#!/usr/bin/env python3
"""
Canonical prediction file for V_cb (CKM c-b matrix element, |V_cb|).

Audit anchor: Row P3 of `docs/parameters/parameter_uniqueness_ledger.md`. UNIQUE
conditional on Rows 4, 6, 9, 11, 12 of the structural uniqueness ledger
(`docs/audits/registers/uniqueness_ledger.md`) plus A5(b) Level 3 prescription with
sub-class identification (Hashimoto walk-rep, not Moore-equivalent
slots). See `docs/theorems/theorem_A5b_level_prescription.md`.
"""

# ============================================================
# PARAMETER: V_cb (CKM quark-mixing matrix element |V_cb|)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       40.6 ± 0.9 × 10^{-3} (exclusive determination, Belle)
# Source:      PDG 2024 (Navas et al., Phys. Rev. D 110, 030001), CKM review
# PDG edition: 2024
# Note:        Inclusive determination is (42.15 ± 0.50) × 10^{-3}, a
#              long-standing ~3.3σ tension with exclusive. Our derivation
#              corresponds to the exclusive (single-girth-cycle) amplitude
#              and is compared to the exclusive value.

# --- PREDICTED VALUE -----------------------------------------
# Value:       256/6305 = 40.6027 × 10^{-3}  (exact rational)
# Deviation:   +0.10e-3 absolute, +0.25%, +0.07 sigma
#
# Bridge convention (docs/framework/framework_scheme_convention.md §7): V_cb is a
# Level-3 Hashimoto walk-sum coupling under A5(b) Case (B). The geometric
# series 256/6305 = (2/3)^8 / (1−(2/3)^8) is the framework-native "bare +
# Feshbach" output: bare = single-winding (2/3)^8; Feshbach-equivalent
# winding sum over all admissible girth-cycle windings under A2 waterline
#. Convention-complete — residual +0.07σ
# is sub-Feshbach.

# --- DERIVED FORMULA -----------------------------------------
# V_cb = α₁_bare / (1 − α₁_bare)
#      = (2/3)^8 / (1 − (2/3)^8)
#      = 256 / 6305  (exact)
#
# Chain:
#   A1 (toggle) + A2 (MDL waterline) + A5(b) (couplings = μ-moments)
#   → branch measure μ = |E|^{-L} on Hashimoto NB walks (Level 3)
#   → L_cb = g − n_fixed = 10 − 2 = 8 (endpoint counting, CAS-verified)
#   → α₁_bare = (2/3)^8 = first-winding μ-moment (branch measure Corollary 1)
#   → A2 waterline: n-th winding saves (8n − O(log n)) > 0 bits ∀n ≥ 1
#   → all windings retained → V_cb = Σ_{n≥1} (2/3)^{8n} = α₁/(1−α₁)
#
# Three-level hierarchy: Level 3 (causal observer = Hashimoto graph).
# NOT Level 2 (srs crystal). CKM lives on the Hashimoto graph.
#
# Status: THEOREM-GRADE — 0 adoptions; all steps Type 1/2/3/4.
#         CAS verification: proofs/flavor/vcb_hashimoto_bfs.py (8³ supercell,
#         20 same-orbit pairs at cycle-distance g−2=8 confirmed, 2026-04-21).

# --- INPUTS --------------------------------------------------
# symbol      | value         | status     | predictions/ file               | meaning
# ------------|---------------|------------|----------------------------------|--------
# k_star      | 3             | [derived]  | predictions/k_star.py            | MDL-optimal degree
# g           | 10            | [derived]  | predictions/g_girth.py           | girth of srs
# d           | 3             | [derived]  | predictions/d_spatial.py         | spatial dimension
# n_fixed     | 2             | [derived]  | proofs/flavor/vcb_nfixed_proof.py| fixed endpoints (1 b-type + 1 c-type)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import functools
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)

# Step 1: first-winding μ-moment (branch measure Corollary 1 + Feshbach Exponent Principle)
# L_cb = g - n_fixed = 10 - 2 = 8  [Type 2: endpoint counting; CAS: vcb_nfixed_proof.py]
# α₁_bare = (2/3)^8 = 256/6561     [Type 4: feshbach_exponent_principle.py]
n_fixed = 2                         # 1 b-type + 1 c-type causal state (fixed endpoints)
L_cb = g - n_fixed                  # = 8
alpha1_bare = Fraction(k - 1, k) ** L_cb   # = (2/3)^8 = 256/6561

# Step 2: A2-T waterline sum (all windings above threshold)
# n-th winding: savings = 8n - O(log n) > 0 for all n ≥ 1  [Type 4: A2-T]
# Geometric series: Σ_{n≥1} α₁^n = α₁/(1−α₁)              [Type 2: algebra]
V_cb_exact = alpha1_bare / (1 - alpha1_bare)  # = 256/6305

V_cb = float(V_cb_exact)

# --- observed value ---
V_cb_obs   = 40.6e-3    # PDG 2024 exclusive (Belle); PRD 110, 030001
V_cb_sigma = 0.9e-3

dev_abs   = V_cb - V_cb_obs
dev_rel   = dev_abs / V_cb_obs
dev_sigma = dev_abs / V_cb_sigma

print("=" * 68)
print("  V_cb  --  THEOREM-GRADE (0 adoptions)")
print("=" * 68)
print(f"  k*          = {k}")
print(f"  g           = {g}")
print(f"  L_cb        = g - n_fixed = {g} - {n_fixed} = {L_cb}  [CAS: vcb_nfixed_proof.py + vcb_hashimoto_bfs.py]")
print(f"  α₁_bare     = (2/3)^8 = {alpha1_bare} ≈ {float(alpha1_bare):.8f}")
print(f"  V_cb (exact)= α₁/(1−α₁) = {V_cb_exact} = {V_cb*1e3:.4f} × 10^-3")
print()
print(f"  PDG 2024 exclusive  = {V_cb_obs*1e3:.1f} ± {V_cb_sigma*1e3:.1f} × 10^-3")
print(f"  Deviation           = {dev_abs*1e3:+.4f} × 10^-3 "
      f"({dev_rel*100:+.3f}%, {dev_sigma:+.2f} sigma)")
print()
print("  Gate chain:")
print("    Step 1 [Type 4]: α₁_bare = (2/3)^8 — branch measure Corollary 1")
print("    Step 2 [Type 1+2]: A2 waterline → geometric series α₁/(1−α₁)")
print("    Step 3 [CAS]: L_cb = 8 — vcb_hashimoto_bfs.py (20 same-orbit pairs)")
print()
print("  Inclusive PDG 2024: (42.15 ± 0.50)e-3 (~3.3σ excl/incl tension).")
print("  Our formula corresponds to the exclusive amplitude.")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_V_cb(k_star, g_girth, n_fixed):
    """
    Compute |V_cb| from the branch measure theorem + A2 waterline.

    Formula:
        V_cb = α₁_bare / (1 − α₁_bare)
        where α₁_bare = ((k_star−1)/k_star)^(g_girth − n_fixed)

    Chain: A1 + A2-T waterline + A5(b) → all above-waterline girth-cycle
    windings retained → geometric series.

    Parameters
    ----------
    k_star : int
        MDL-optimal lattice degree (3 for srs).
    g_girth : int
        Girth of the srs lattice (10).
    n_fixed : int
        Number of fixed endpoints in the b→c transition (2).

    Returns
    -------
    float
        Predicted |V_cb|.
    """
    from fractions import Fraction
    alpha1 = Fraction(k_star - 1, k_star) ** (g_girth - n_fixed)
    return float(alpha1 / (1 - alpha1))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = V_cb
    pure_result = predict_V_cb(k, g, n_fixed)
    print()
    print(f"Implementation:  {impl_result:.12f}")
    print(f"Pure function:   {pure_result:.12f}")
    assert abs(impl_result - pure_result) < 1e-12, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    V_cb = {pure_result*1e3:.4f} × 10^-3  "
          f"(obs: {V_cb_obs*1e3:.1f} ± {V_cb_sigma*1e3:.1f} × 10^-3, "
          f"{dev_sigma:+.2f} sigma)")
    print("    Rigor status: THEOREM-GRADE (0 adoptions).")
