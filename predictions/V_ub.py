#!/usr/bin/env python3
"""
Canonical prediction file for V_ub (CKM u-b matrix element, |V_ub|).

STATUS UNDER PARAMETER LINTER (2026-04-30 graduation, propagated 2026-05-02):
UNIQUE-THEOREM-GRADE for amplitude form; labeling data-anchored, non-blocking.
Clause 7 PASS-CITED; Clause 8 PASS at −0.26σ on PDG combined.

Audit anchor: Row P14 of `docs/parameters/parameter_uniqueness_ledger.md`.

    V_ub = Σ_{m ≥ 2} (2/3)^{6m+2} / (1 − (2/3)^{6m+2}) ≈ 3.767 × 10⁻³

The amplitude form is theorem-grade via M1 twisted-walker Bloch matrix-element
closure (`proofs/foundations/m1_twisted_walker_v_cb_v_ub.py` +
`m1_n_orbit_3orbit_basis.py`, commit 753f4cf, 2026-04-30): the squared-modulus
rule

    |⟨g_(L mod 3) | T^L | g_0⟩|² / 3^L = (2/3)^L = α_L      at L = 6m+2

(T = B_total · C_36 twisted walker on N-orbit cyclic 3-orbit basis) reproduces
α_m = (2/3)^L exactly. Combined with H(srs) multi-cycle host topology giving
L_eff(m) = 6m+2, this fixes V_cb ↔ m=1 host and V_ub ↔ Σ_{m≥2} hosts at
theorem grade. A2 waterline retains all windings → α_m/(1−α_m) per host
class; the sum over m ≥ 2 converges geometrically.

Labeling layer is OTHER-SMUGGLE residue, NON-BLOCKING for predictive content
per the (Z/2)^3 Angle D verdict
+ Z3-mass-order verdict,
commit e5ef667, 2026-04-30: under PS-spinor-weight relabeling (a) Γ_7 sign /
L↔R, (b) Y sign / lepton↔quark, (c) T_L↔T_R, all 77 prediction values are
invariant; only (PDG name → value) pairings shift. Therefore the labeling
residue is a global naming convention pinned by empirical anchoring of
names — not a predictive gap.

History:
  - Pre-2026-04-25: B3 + Type A sector-universality reading gave V_ub = 0
    (RETIRED when V_cb closed at theorem grade in session 13).
  - 2026-04-25 to 2026-04-28 AM: BLOCKED with sentinel 0 pending substrate-
    Z₃ = generation derivation.
  - 2026-04-28 PM: graduated to STRICT-SOLID THEOREM-GRADE via bridge
    functoriality lemma (RETRACTED 2026-04-29 — three CAS probes refute
    the load-bearing Z_3 holonomy step).
  - 2026-04-29: UNIQUE-THEOREM-GRADE for amplitude (M1); labeling data-anchored non-blocking.
  - 2026-04-30: graduated to UNIQUE-THEOREM-GRADE for amplitude (M1 twisted
    walker — different mechanism than the retracted bridge functoriality)
    + labeling reframed data-anchored / non-blocking via Angle D +
    Z3-mass-order. ADOPTED-A5b-Sub3 obsolete; replaced by M1 amplitude
    theorem.
  - 2026-05-02: this file's status banners propagated to reflect the
    2026-04-30 graduation (file was previously stale).
"""

# ============================================================
# PARAMETER: V_ub (CKM u-b matrix element, |V_ub|)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       |V_ub| = 0.00369 ± 0.00011  (PDG 2024 exclusive)
#              |V_ub| = 0.00413 ± 0.00015  (PDG 2024 inclusive)
#              |V_ub| ≈ 0.00382 ± 0.00020  (PDG 2024 combined exc+inc;
#                                           ~3σ exc/inc tension)
# Source:      PDG 2024 Review of Particle Physics, CKM review.
# PDG edition: 2024.

# --- PREDICTED VALUE -----------------------------------------
# Value:       |V_ub| = 3.767 × 10⁻³  (multi-cycle walk-rep sum)
# Deviation:   −0.26σ from PDG combined exc+inc; +0.70σ from PDG exclusive;
#              −2.42σ from PDG inclusive (within the well-known exc/inc
#              experimental tension band).
# Status:      UNIQUE-THEOREM-GRADE for amplitude form via M1 twisted walker
#              (commit 753f4cf, 2026-04-30); labeling data-anchored,
#              non-blocking via Angle D + Z3-mass-order (commit e5ef667).
#              Clause 8 PASS at −0.26σ; systematic floor zero (pure structural).
#
# Bridge convention (docs/framework/framework_scheme_convention.md §7): V_ub is a
# Level-3 Hashimoto walk-sum coupling under A5(b) Case (B), parallel to
# V_cb. The amplitude α_m/(1−α_m) per host class is the framework-native
# "bare + Feshbach" sum: bare = single-winding (2/3)^{L_eff(m)}; Feshbach-
# equivalent winding sum over all admissible girth-cycle windings under
# A2 waterline. Convention-complete at the amplitude level; the residual
# (Z/2)^3 labeling freedom does not shift any prediction value (Angle D).

# --- DERIVED FORMULA -----------------------------------------
# V_ub = Σ_{m ≥ 2} α_m / (1 − α_m)
#
# where for m girth-10 cycles glued by m−1 2-edge seams (s_seam = 2):
#   L_cycle(m) = m·g − 2(m−1)·s = 6m + 4
#   L_eff(m)   = L_cycle(m) − n_fixed = 6m + 2
#   α_m        = ((k*−1)/k*)^{L_eff(m)} = (2/3)^{6m+2}
#
# Chain:
#   A1 (toggle) + A2 (MDL waterline) + A5(b) Case B
#     → multi-cycle host topology on H(srs):
#         m=1: single girth-10 cycle    (V_cb's host class)
#         m≥2: composite hosts          (V_ub's host class)
#     → branch measure (Theorem of multiway branch measure §3+§4)
#     → Feshbach exponent principle: α_m = (2/3)^{6m+2} per winding
#     → A2 waterline: all windings retained → geometric series α_m/(1−α_m)
#     → Sum over multi-cycle topologies (m = 2, 3, 4, …): geometrically
#       convergent; truncation at m_max = 10 saturates to ~14 digits.
#
# The amplitude assignment "V_cb ↔ m=1 host; V_ub ↔ Σ_{m≥2} hosts" is
# theorem-grade via the M1 twisted-walker squared-amplitude rule
# (proofs/foundations/m1_twisted_walker_v_cb_v_ub.py + m1_n_orbit_3orbit_basis.py,
# commit 753f4cf, 2026-04-30):
#   |⟨g_(L mod 3) | T^L | g_0⟩|² / 3^L = (2/3)^L = α_L  at L = 6m+2
# where T = B_total · C_36 is the twisted walker on N-orbit cyclic 3-orbit
# basis. This reproduces α_m exactly, fixing the (m, V_ij) correspondence
# at theorem grade.
#
# Predecessor mechanisms (now obsolete):
#   - Bridge functoriality lemma (Z_3^m holonomy): RETRACTED 2026-04-29.
#     - Z_3 holonomy refuted: flat connection theorem
#       (proofs/flavor/z3_holonomy_cycles.py).
#     - Pinning topology refuted: higher-m probe
#       (proofs/flavor/vub_bridge_higher_m_pinning_probe.py).
#     - Combinatorial Z_3 refuted: 50/50 split classifier
#       (proofs/flavor/vub_bridge_z3_shift_classifier.py).
#   - ADOPTED-A5b-Sub3 (un-graduated adoption): obsolete, replaced by M1.
#
# Residual labeling residue: the (Z/2)^3 PS-spinor-weight relabeling
# freedom (Γ_7 sign, Y sign, T_L↔T_R) does NOT shift any prediction value
# (Angle D verdict 2026-04-30, all 77 prediction files invariant under
# the (Z/2)^3 group action). Only (PDG name → value) pairings shift —
# that is, the labeling residue is empirical anchoring of names, not a
# predictive gap.
#
# CAS verification: proofs/flavor/vub_multicycle_sum.py emits
# 3.7670e-3 (m_max=10 truncation; converged to 14 digits).

# --- INPUTS --------------------------------------------------
# symbol      | value          | status     | predictions/ file               | meaning
# ------------|----------------|------------|----------------------------------|--------
# k_star      | 3              | [derived]  | predictions/k_star.py            | MDL-optimal degree
# g           | 10             | [derived]  | predictions/g_girth.py           | girth of srs
# d           | 3              | [derived]  | predictions/d_spatial.py         | spatial dimension
# n_fixed     | 2              | [derived]  | proofs/flavor/vcb_nfixed_proof.py| fixed endpoints (1 b-type + 1 u-type)
# s_seam      | 2              | [derived]  | hashimoto_16cycle_decomposition  | seam length (CAS-verified m=2 case)
# m_max       | 10             | [scope]    | (truncation; series converges)   | truncates at ~14 digits
# M1 amplitude theorem                      | proofs/foundations/m1_twisted_walker_v_cb_v_ub.py | fixes V_cb↔m=1, V_ub↔Σ_{m≥2}

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
s_seam = 2          # CAS-verified seam length on m=2 hosts
n_fixed = 2         # 1 b-type + 1 u-type causal state pinning
m_max = 10          # truncation; series converges to ~14 digits at this depth


def _alpha_m(k_star, g_girth, s_seam, n_fixed, m):
    """Per-winding amplitude on the m-cycle host (Feshbach exponent principle)."""
    L_cycle = m * g_girth - 2 * (m - 1) * s_seam
    L_eff = L_cycle - n_fixed
    return Fraction(k_star - 1, k_star) ** L_eff


def _v_ub_partial_sum(k_star, g_girth, s_seam, n_fixed, m_max):
    """Multi-cycle walk-rep sum: V_ub = Σ_{m=2}^{m_max} α_m / (1 − α_m)."""
    total = Fraction(0)
    for m in range(2, m_max + 1):
        a = _alpha_m(k_star, g_girth, s_seam, n_fixed, m)
        total += a / (1 - a)
    return total


V_ub_exact = _v_ub_partial_sum(k, g, s_seam, n_fixed, m_max)
V_ub = float(V_ub_exact)

# --- observed values ---
V_ub_obs_excl = 3.69e-3;  V_ub_unc_excl = 0.11e-3
V_ub_obs_incl = 4.13e-3;  V_ub_unc_incl = 0.15e-3
V_ub_obs_comb = 3.82e-3;  V_ub_unc_comb = 0.20e-3   # combined; covers exc/inc tension

dev_abs   = V_ub - V_ub_obs_comb
dev_rel   = dev_abs / V_ub_obs_comb
dev_sigma = dev_abs / V_ub_unc_comb

# Runner-facing canonical aliases (slug = "V_ub"); aliases only.
V_ub_pred  = V_ub
V_ub_obs   = V_ub_obs_comb
V_ub_sigma = V_ub_unc_comb

print("=" * 70)
print("  V_ub  --  UNIQUE-THEOREM-GRADE for amplitude (M1); labeling data-anchored non-blocking")
print("=" * 70)
print(f"  k*       = {k}")
print(f"  g        = {g}")
print(f"  s_seam   = {s_seam}  (CAS-verified for m=2)")
print(f"  n_fixed  = {n_fixed}  (1 b-type + 1 u-type pinning)")
print(f"  m_max    = {m_max}  (truncation; series converges)")
print()
print(f"  V_ub = Σ_{{m=2}}^{m_max} (2/3)^(6m+2) / (1 − (2/3)^(6m+2))")
print(f"       = {float(V_ub_exact):.6e}")
print()
print(f"  PDG 2024 exclusive : {V_ub_obs_excl*1e3:.2f} ± {V_ub_unc_excl*1e3:.2f} × 10⁻³"
      f"  → {((V_ub - V_ub_obs_excl)/V_ub_unc_excl):+.2f}σ")
print(f"  PDG 2024 inclusive : {V_ub_obs_incl*1e3:.2f} ± {V_ub_unc_incl*1e3:.2f} × 10⁻³"
      f"  → {((V_ub - V_ub_obs_incl)/V_ub_unc_incl):+.2f}σ")
print(f"  PDG 2024 combined  : {V_ub_obs_comb*1e3:.2f} ± {V_ub_unc_comb*1e3:.2f} × 10⁻³"
      f"  → {dev_sigma:+.2f}σ")
print()
print("  Status: UNIQUE-THEOREM-GRADE for amplitude (M1); labeling data-anchored non-blocking.")
print("  Amplitude form theorem-grade via M1 twisted walker (commit 753f4cf, 2026-04-30):")
print("    |⟨g_(L mod 3) | T^L | g_0⟩|² / 3^L = (2/3)^L = α_L  at L = 6m+2")
print("  Labeling residue (Z/2)^3 PS-spinor freedom shifts no prediction value (Angle D verdict).")
print()
print("  Per-m breakdown (leading + corrections):")
for m in range(2, 6):
    a = _alpha_m(k, g, s_seam, n_fixed, m)
    contrib = float(a / (1 - a))
    pct = contrib / V_ub * 100
    print(f"    m={m}: L_eff={6*m+2:>2d}, V_m = {contrib:.6e}  ({pct:>6.3f}% of total)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_V_ub(k_star, g_girth, s_seam, n_fixed, m_max):
    """
    Compute |V_ub| from the multi-cycle walk-rep sum on H(srs).

    Formula:
        V_ub = Σ_{m=2}^{m_max} α_m / (1 − α_m)
        α_m  = ((k_star−1)/k_star)^{L_eff(m)}
        L_eff(m) = m·g_girth − 2(m−1)·s_seam − n_fixed = 6m+2 for srs

    Status: UNIQUE-THEOREM-GRADE for amplitude form via M1 twisted walker
    (commit 753f4cf, 2026-04-30); labeling layer data-anchored, non-blocking
    via Angle D + Z3-mass-order (commit e5ef667). The amplitude formula
    passes the parameter_linter gate; the (m, V_ij) correspondence
    "V_cb ↔ m=1, V_ub ↔ Σ_{m≥2}" is fixed at theorem grade by the M1
    squared-amplitude rule.

    Parameters
    ----------
    k_star : int
        MDL-optimal lattice degree (3 for srs).
    g_girth : int
        Girth of the srs lattice (10).
    s_seam : int
        Seam length between consecutive girth cycles in multi-cycle host (2).
    n_fixed : int
        Number of fixed endpoint causal states (2).
    m_max : int
        Truncation index for the multi-cycle sum. Series converges
        geometrically; m_max = 10 saturates to ~14 digits for k_star=3.

    Returns
    -------
    float
        Predicted |V_ub|.
    """
    from fractions import Fraction
    total = Fraction(0)
    for m in range(2, m_max + 1):
        L_cycle = m * g_girth - 2 * (m - 1) * s_seam
        L_eff = L_cycle - n_fixed
        alpha = Fraction(k_star - 1, k_star) ** L_eff
        total += alpha / (1 - alpha)
    return float(total)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = V_ub
    pure_result = predict_V_ub(k, g, s_seam, n_fixed, m_max)
    print()
    print(f"Implementation: {impl_result:.12e}")
    print(f"Pure function:  {pure_result:.12e}")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    V_ub = {pure_result*1e3:.4f} × 10⁻³  "
          f"(PDG combined: {V_ub_obs_comb*1e3:.2f} ± {V_ub_unc_comb*1e3:.2f} × 10⁻³, "
          f"{dev_sigma:+.2f}σ)")
    print("    Rigor status: UNIQUE-THEOREM-GRADE for amplitude (M1); labeling data-anchored non-blocking.")
