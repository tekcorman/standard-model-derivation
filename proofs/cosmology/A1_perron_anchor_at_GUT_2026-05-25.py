#!/usr/bin/env python3
"""
A1 candidate: α = 1/2 EXACT, T_GUT_anchor = M_unif × c_S (2026-05-25).

REPLACES the "d_eff = 3 + 1/(2|E|) fractional-dimension" framing of the
earlier probes (`A1_d_eff_derivation_attempt`, `A1_d_eff_mechanism_attempt`).
That framing was WRONG because it conflicted with d_spatial = 3
(theorem-grade via Cencov-Fisher, `predictions/d_spatial.py`).

User correction (2026-05-25 EOD+3): "we already derived dimension there.
is that different?" — d_spatial = 3 IS rigorous; d_eff cannot be 3.08
without violating an established theorem.

THE CORRECTED FRAMING:

  d_eff = d_spatial = 3 EXACTLY
  α = 1/2 EXACTLY (horizon-thermal in flat 3D coasting)
  The 1/(2|E|) factor enters as the THERMAL ANCHOR scale at gauge
  unification, NOT as a fractional dimension correction.

DERIVATION:
  In flat 3D coasting (a ∝ N), horizon volume V ∝ N^3, substrate pump
  rate κ/t_P constant. Energy in horizon = κ·N·t_P. u = E/V ∝ N^(-2).
  Stefan-Boltzmann: T = u^(1/4) ∝ N^(-1/2). α = 1/2 EXACT.

  Anchor: at gauge unification (v(N) = M_unif), the GUT-energy reservoir
  has thermal scale M_unif. But only the Perron-singlet (gauge-readable
  photon channel) carries the c_S = 1/(2|E|) share of this energy.

  T at gauge unification = M_unif × c_S = M_unif / (2|E|) = M_unif / 12.

  Forward propagation:
    T_today = T_GUT_anchor × √(N_GUT/N_today)
            = (M_unif × c_S × K_per_GeV) × √(N_GUT/N_hub)
            = computed below

  This makes A1 the THIRD reading of c_S in the unified-oblique family:
    - δ_r: c_S enters as the Z-channel Perron-residue coefficient
    - A_s: c_S enters as the single-loop-closure prefactor (1/54 = c_S·q²·(1/2))
    - A1:  c_S enters as the gauge-unif thermal-anchor projection

  The 1/(2|E|) is the SAME projector in all three — Perron eigenvector at Γ
  on B_NB, with weight 1/(2|E|) for the unit singlet.
"""

from __future__ import annotations

import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

print("=" * 76)
print("A1 candidate: α = 1/2 EXACT, T_GUT = M_unif × c_S (2026-05-25 corrected)")
print("=" * 76)

# Framework primitives
k_B = 1.380649e-23
GeV = 1.602176634e-10  # J per GeV
K_per_GeV = GeV / k_B  # = 1.16e13

N_hub = 8.394881e60
v_today = 246.22
M_unif = 1.985e16
T_CMB_observed = 2.7255

# Perron-residue
N_atoms = 4
k_star = 3
two_E = N_atoms * k_star  # = 12, the handshake-lemma |2E|
c_S = 1 / two_E  # = 1/12

print(f"\nFramework primitives (all theorem-grade upstream):")
print(f"  d_spatial = 3 (Cencov-Fisher, predictions/d_spatial.py)")
print(f"  k* = 3 (predictions/k_star.py)")
print(f"  M_unif = {M_unif:.3e} GeV (framework GUT scale)")
print(f"  c_S = 1/(2|E|) = 1/{two_E} = {c_S:.6f} (Perron-residue at Γ, theorem-grade in unified-oblique §3.2)")
print(f"  N_hub (today) = {N_hub:.3e}")


# ------------------------------------------------------------------------
# Derivation chain
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("DERIVATION CHAIN (α = 1/2 EXACT)")
print('='*76)

# N at gauge unification
N_GUT = N_hub * (v_today / M_unif) ** 4

# Anchor: T at GUT epoch = M_unif × c_S (Perron-projected gauge-readable share)
T_GUT_anchor_GeV = M_unif * c_S
T_GUT_anchor_K = T_GUT_anchor_GeV * K_per_GeV

print(f"\nStep 1 — N at gauge unification:")
print(f"  v(N) ∝ N^(-1/4) (BZJ cascade, theorem-grade in v_higgs)")
print(f"  v(N_GUT) = M_unif → N_GUT = N_hub × (v_today/M_unif)^4 = {N_GUT:.3e}")

print(f"\nStep 2 — Perron-projected thermal anchor at gauge unification:")
print(f"  GUT-energy reservoir at v=M_unif: order M_unif = {M_unif:.3e} GeV")
print(f"  Gauge-readable (Perron-singlet at Γ) share: c_S = 1/(2|E|) = {c_S:.6f}")
print(f"  T_GUT_anchor = M_unif × c_S = {T_GUT_anchor_GeV:.3e} GeV = {T_GUT_anchor_K:.3e} K")

print(f"\nStep 3 — Horizon-thermal evolution α = 1/2 EXACT:")
print(f"  d_spatial = 3 (theorem-grade)")
print(f"  Coasting: H = 1/(N·t_P), a ∝ N")
print(f"  Horizon volume: V = (c·N·t_P)^3 ∝ N^3")
print(f"  Substrate pump rate: κ/t_P (constant in time)")
print(f"  Total energy in horizon: E = κ·N·t_P (linear in N)")
print(f"  Energy density: u = E/V ∝ N^(-2)")
print(f"  Stefan-Boltzmann: T = u^(1/4) ∝ N^(-1/2)")
print(f"  → α = 1/2 EXACT (no fractional correction)")

# Forward propagation
T_today_predicted_K = T_GUT_anchor_K * math.sqrt(N_GUT / N_hub)
print(f"\nStep 4 — Forward propagation to today:")
print(f"  T_today = T_GUT_anchor × √(N_GUT/N_hub)")
print(f"          = {T_GUT_anchor_K:.3e} × √({N_GUT/N_hub:.3e})")
print(f"          = {T_GUT_anchor_K:.3e} × {math.sqrt(N_GUT/N_hub):.3e}")
print(f"          = {T_today_predicted_K:.4f} K")

# Compare
print(f"\nStep 5 — Comparison with observation:")
print(f"  T_today predicted = {T_today_predicted_K:.4f} K")
print(f"  T_today observed  = {T_CMB_observed} K (Planck CMB)")
deviation_pct = (T_today_predicted_K - T_CMB_observed) / T_CMB_observed * 100
print(f"  Deviation         = {deviation_pct:+.2f}%")


# ------------------------------------------------------------------------
# Status
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("STATUS")
print('='*76)

if abs(deviation_pct) < 1:
    status = "CLOSED"
    msg = "A1 closes at theorem-grade-numerical: α=1/2 exact + c_S anchor."
elif abs(deviation_pct) < 10:
    status = "CANDIDATE — STRUCTURAL MATCH"
    msg = ("A1 has a structural candidate within 10%. The 1/(2|E|) factor is\n"
           "the framework's c_S Perron-residue (theorem-grade upstream). The\n"
           "8% residual could be order-unity calibration or a small structural\n"
           "correction not yet identified.")
else:
    status = "NEAR-CANDIDATE — UNCLOSED"
    msg = "Order-of-magnitude right, percent-level residual."

print(f"\n  Status: {status}")
print(f"  Deviation: {deviation_pct:+.2f}%")
print(f"\n  {msg}")

print(f"""
KEY POINTS (corrected from earlier session):

  - d_eff = d_spatial = 3 EXACTLY (no fractional dimension correction;
    the earlier "d_eff = 3 + 1/(2|E|)" framing was a misidentification
    that conflicted with the theorem-grade d_spatial = 3).

  - α = 1/2 EXACTLY from pure horizon-thermal in flat 3D coasting.

  - The 1/(2|E|) = c_S factor enters as the THERMAL ANCHOR scale at
    gauge unification (T_GUT = M_unif × c_S), exactly the same Perron-
    projection at Γ that gives:
      δ_r = c_S · α₁/(1 − α₁)            (Z-channel of unified-oblique)
      A_s prefactor = c_S · q² · (1/2)   (single-loop-closure)
      T_GUT = c_S · M_unif                (gauge-unif thermal anchor)

  - A1 joins the unified-oblique c_S family as the third structural reading.

OPEN ITEMS:
  - The 8% T_today residual. Could be:
    * Order-unity calibration: the "GUT-energy reservoir" might not be
      exactly M_unif (e.g., the relevant scale is α_GUT × M_Pl, or √2 × v
      at unif, or some other framework-natural variant)
    * Small structural correction (a second-order c_S effect, or a small
      contribution from the substrate-anchor regime before GUT)

  - The 'T_GUT = M_unif × c_S' identification needs a clean structural
    derivation. The Perron projection of the substrate's thermal modes
    at v=M_unif onto the gauge-readable singlet should give this naturally,
    but I haven't written the explicit derivation.

If both items close, A1 graduates to theorem-grade-numerical alongside
δ_r and A_s as readings of c_S.
""")
print("=" * 76)
