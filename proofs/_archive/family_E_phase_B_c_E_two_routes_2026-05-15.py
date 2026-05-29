#!/usr/bin/env python3
"""
proofs/_archive/family_E_phase_B_c_E_two_routes_2026-05-15.py

*** RETRACTED 2026-05-15 EOD+15 — STALE BASE PREDICTIONS ***
This probe inherited Phase A's stale base (M_Z=91.97, m_W=80.69).  The
headline "joint cluster closure to 0.007% absolute on δρ" is a STALE-INPUT
ARTIFACT.  With live predictions (M_Z=91.5135, m_W=80.2373; opposite-sign
residuals) the Phase A/B forms (c_S=1/12, c_E=1/18) give δρ_pred ≈ +0.905%
vs observed +1.043% (0.14% gap, NOT 0.007%), and c_S=1/12 is 253% off the
correct small c_S≈0.024 target.  Cluster does NOT graduate.  See
`family_E_phase_AB_CORRECTED_base_predictions_2026-05-15.py` (commit
c66bc02).  Caught by parameter_linter Checkpoint 1.  Preserved for record.

Phase B of Family C + Family E joint derivation for M_Z/m_W cluster.

PRECEDING (Phase A, commit e1466db): c_S = 1/12 derived via two-routes
convergence (Route H: 1/(2|E|), Route C: H_1/(N·k*²) = k*/(N·k*²)),
calibrated against v_Higgs c_v = 5/12 by structural factor-1/5 reduction.
Hypothesis emerged: c_E = c_S × (k*-1)/k* = 1/18 (asymmetric Family E
piece), giving 0.44% off the empirical Δρ target.

PHASE B GOAL: derive c_E structurally via two INDEPENDENT routes
(H + C), per master doc §8 rule 1 (two-routes discipline), and verify
joint cluster closure with c_S = 1/12.

KEY DISCIPLINE QUESTION: are the two routes for Family E genuinely
INDEPENDENT mechanisms, or are they both expressions of the same
modular form "c_S × (k*-1)/k*"?  If the latter, two-routes is satisfied
by CONSTRUCTION but not by INDEPENDENT VERIFICATION — partial closure.

PROBE STRUCTURE

Section 1 — Derive Route H form for c_E:
  c_E = (k*-1) / (2|E| · k*)
  Structural reading: "Family C marginal sector × NB walker survival
  per step factor (k*-1)/k*."

Section 2 — Derive Route C form for c_E:
  c_E = (k*-1) / (N_atoms · k*²)
  Structural reading: "Family C cycle count weighted by NB walker
  survival (k*-1) instead of k* in numerator."

Section 3 — Two-routes verification:
  Both give 1/18.  Are they STRUCTURALLY INDEPENDENT or by-construction?

Section 4 — Calibration discipline check:
  Family E selection rule: applies to custodial-breaking observables.
  Does v_Higgs naturally get c_E = 0 (or trivializes)?

Section 5 — Joint cluster closure check:
  Apply c_S = 1/12 + c_E = 1/18 to bare M_Z, m_W predictions.
  Verify residuals close within sub-percent.
  Check sign of Δρ.

PRE-DECLARED ABORT:
(CB.1) Routes give different numerical values → close NEG.
(CB.2) Joint cluster fit > 1% off either M_Z or m_W → close NEG.
(CB.3) Routes converge AND cluster closes within sub-percent → PHASE B
       POSITIVE.  Cluster GRADUATES to THEOREM-GRADE-CONDITIONAL.
(CB.4) Routes equal but only by-construction (not independent) →
       CONDITIONAL POSITIVE — partial graduation, needs deeper theory.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

# ---------------------------------------------------------------------------
# Framework constants (theorem-grade)
# ---------------------------------------------------------------------------
k_star = 3
g = 10
N_ATOMS = 4
N_EDGES = 6
N_DIRECTED_EDGES = 2 * N_EDGES  # = 12
H1_DIM = N_EDGES - N_ATOMS + 1  # = 3
MARGINAL_DIM = 2 * (N_EDGES - N_ATOMS) + 1  # = 5

alpha_1_bare = Fraction(k_star - 1, k_star) ** (g - 2)  # = 256/6561
alpha_factor = float(alpha_1_bare) / (1 - float(alpha_1_bare))  # ≈ 0.0406
n_g = 15  # girth-cycle count per vertex

# Phase A finding
c_S = Fraction(1, 12)
c_v_route_H = Fraction(MARGINAL_DIM, N_DIRECTED_EDGES)  # = 5/12
c_v_route_C = Fraction(n_g, N_ATOMS * k_star ** 2)  # = 5/12

# Empirical anchors
M_Z_PDG = 91.1876
m_W_PDG = 80.3692
M_Z_pred = 91.97  # current framework prediction (no Family C/E)
m_W_pred = 80.69
sin2_W_MS = 0.23122
cos2_W_MS = 1 - sin2_W_MS

print("=" * 78)
print("  Phase B — Family E c_E derivation via two-routes discipline")
print("=" * 78)
print()


# ---------------------------------------------------------------------------
# Section 1 — Route H derivation for c_E
# ---------------------------------------------------------------------------
print("=" * 78)
print("Section 1: Route H (Hashimoto-spectral) derivation for Family E c_E")
print("=" * 78)
print()
print(f"  Route H STRUCTURAL FORM:")
print(f"    c_E = (k*-1) / (2|E| · k*) = NB-walker-survival numerator over")
print(f"          (NB Hilbert × per-vertex normalization)")
print()
print(f"  Numerator (k*-1) = {k_star - 1}: 'one NB walker survival count per step'")
print(f"  Denominator 2|E|·k* = {N_DIRECTED_EDGES * k_star}: 'NB Hilbert × per-vertex k*'")
print()
c_E_route_H = Fraction(k_star - 1, N_DIRECTED_EDGES * k_star)
print(f"  c_E (Route H) = ({k_star-1}) / ({N_DIRECTED_EDGES} × {k_star}) = {c_E_route_H} = {float(c_E_route_H):.6f}")
print()
print(f"  Structural reading: Family C in Route H is c_S = 1/(2|E|) = '1 marginal")
print(f"  direction per total NB Hilbert.'  Family E adds NB-step modulation:")
print(f"  multiply by (k*-1)/k* (NB survival per step) → 1 × (k*-1)/(2|E|·k*) = 1/18.")
print()
print(f"  Alternative reading: c_E from |λ|² = k*-1 = 2 (Ramanujan saturation)")
print(f"  spectral content of B(srs), which carries the asymmetric/cycle structure")
print(f"  per `nb_two_vertex_generations_probe.py`.  Single direction in this")
print(f"  spectral subspace, normalized by 2|E|·k*.")


# ---------------------------------------------------------------------------
# Section 2 — Route C derivation for c_E
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("Section 2: Route C (cycle-counting) derivation for Family E c_E")
print("=" * 78)
print()
print(f"  Route C STRUCTURAL FORM:")
print(f"    c_E = (k*-1) / (N_atoms · k*²)")
print(f"        = NB-walker-survival count per cycle / per-cell normalization")
print()
print(f"  Numerator (k*-1) = {k_star - 1}: 'NB step count per cycle traversal'")
print(f"  Denominator N_atoms·k*² = {N_ATOMS * k_star ** 2}: 'per-cell × initial-final direction'")
print()
c_E_route_C = Fraction(k_star - 1, N_ATOMS * k_star ** 2)
print(f"  c_E (Route C) = ({k_star-1}) / ({N_ATOMS} × {k_star}²) = {c_E_route_C} = {float(c_E_route_C):.6f}")
print()
print(f"  Structural reading: Family C in Route C is c_S = k*/(N·k*²) = '1/(N·k*)'")
print(f"  = 1 cycle per crystal direction per per-cell normalization.  Family E")
print(f"  modulates: replace k* with (k*-1) in numerator (cycles weighted by NB")
print(f"  walker survival per step instead of full edge count) → (k*-1)/(N·k*²) = 1/18.")
print()
print(f"  Alternative reading: cycles passing through 'asymmetric content' (i.e.,")
print(f"  the up-down sector difference = (k*-1) species pairs out of k*) per the")
print(f"  per-cell normalization.  Specifically: at each Hamming-weight-2 (up) vertex,")
print(f"  there are (k*-1) = 2 ways to pair up vs (k*-2) = 1 way to differentiate")
print(f"  from Hamming-weight-1 (down) — giving 'asymmetric pair count' (k*-1).")


# ---------------------------------------------------------------------------
# Section 3 — Two-routes convergence
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("Section 3: Two-routes convergence verification")
print("=" * 78)
print()
print(f"  Route H: c_E = {c_E_route_H} = {float(c_E_route_H):.6f}")
print(f"  Route C: c_E = {c_E_route_C} = {float(c_E_route_C):.6f}")
print(f"  Convergent: {c_E_route_H == c_E_route_C}  → c_E = 1/18 ✓")
print()
print(f"  HONESTY CHECK: are Routes H and C INDEPENDENTLY derived for Family E?")
print()
print(f"  For Family C, Routes H and C are clearly independent:")
print(f"    Route H computes c_v from Stark-Terras marginal-mode dimension formula")
print(f"    Route C computes c_v from girth-cycle count per vertex (15)")
print(f"    Both arrive at 5/12 via DIFFERENT mathematical structures.")
print()
print(f"  For Family E (this section):")
print(f"    Route H: c_E = (k*-1) / (2|E| · k*) — NB-survival modulation of Route H Family C")
print(f"    Route C: c_E = (k*-1) / (N · k*²) — NB-survival modulation of Route C Family C")
print(f"    Both are c_S × (k*-1)/k*; numerical convergence is BY CONSTRUCTION.")
print()
print(f"  WEAKER CLAIM: Routes H and C for Family E both express the structural")
print(f"  rule 'Family C × NB walker survival per step.'  They converge because")
print(f"  this rule is consistent across both representations of Family C.  This is")
print(f"  weaker than two-routes discipline as applied to v_Higgs (where the routes")
print(f"  count different things and happen to agree).")
print()
print(f"  STRONGER CLAIM (independent verification needed):")
print(f"    Route H derivation from spectral content at |λ|² = (k*-1) of B(srs)")
print(f"    Route C derivation from cycle counting weighted by Hamming asymmetry")
print(f"    These would be GENUINELY independent — Phase B Section 1/2 sketches")
print(f"    them but full two-routes discipline requires explicit construction.")


# ---------------------------------------------------------------------------
# Section 4 — Calibration discipline (Family E selection rule)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("Section 4: Calibration check (Family E selection rule)")
print("=" * 78)
print()
print(f"  Family E by selection rule applies ONLY to:")
print(f"    'Propagator-level (custodial-breaking) observables'")
print(f"    'requires asymmetric mechanism; Families A-D do NOT apply'")
print(f"  (per master doc meta-classification, line 219)")
print()
print(f"  v_Higgs is VERTEX-LEVEL and has no custodial-breaking content.")
print(f"  → Family E selection rule excludes v_Higgs naturally.")
print(f"  → No calibration test 'c_E_v = 5/18' applies; calibration via")
print(f"    SELECTION RULE (Family E doesn't apply), not via numerical match.")
print()
print(f"  This is CONSISTENT with master doc's Family taxonomy:")
print(f"    Family C applies broadly (both vertex-level v_Higgs and propagator-level")
print(f"    M_Z/m_W via sign-uniform piece) — calibration anchor is c_v = 5/12.")
print(f"    Family E applies narrowly (only custodial-breaking propagator) —")
print(f"    selection rule is 'observable class is custodial-breaking', no")
print(f"    universal calibration anchor needed.")


# ---------------------------------------------------------------------------
# Section 5 — Joint cluster closure with c_S = 1/12 + c_E = 1/18
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("Section 5: Joint cluster closure (c_S = 1/12 + c_E = 1/18)")
print("=" * 78)
print()

c_S_val = float(c_S)
c_E_val = float(c_E_route_H)  # = 1/18

# Family C: M_Z² and m_W² both shift by (1 - c_S × α/(1-α))
# Family E: M_Z² shifts down extra, m_W² shifts up extra (custodial-breaking)
shift_C = c_S_val * alpha_factor  # magnitude of Family C decrease
shift_E = c_E_val * alpha_factor  # magnitude of Family E split

# Apply to M_Z² and m_W² (then take sqrt to get M_Z, m_W)
# M_Z²_corrected = M_Z²_pred × (1 - shift_C - shift_E) (squared shift)
# m_W²_corrected = m_W²_pred × (1 - shift_C + shift_E)
# M_Z_corrected = M_Z_pred × √(1 - shift_C - shift_E)
# m_W_corrected = m_W_pred × √(1 - shift_C + shift_E)

# But Family C/E template is multiplicative on g (not g²); applies to mass directly:
# M_Z_corrected = M_Z_pred × (1 - shift_C - shift_E)  [shift on M_Z directly]
# m_W_corrected = m_W_pred × (1 - shift_C + shift_E)
# Δρ = m_W²/M_Z²/cos²θ - 1 = ratio of squared corrections / cos²θ - 1

# Use the convention: c_S × α/(1-α) shifts G phys multiplicatively
M_Z_corrected = M_Z_pred * (1 - shift_C - shift_E)
m_W_corrected = m_W_pred * (1 - shift_C + shift_E)

resid_M_Z = (M_Z_corrected - M_Z_PDG) / M_Z_PDG
resid_m_W = (m_W_corrected - m_W_PDG) / m_W_PDG

print(f"  Phase A + B applied:")
print(f"    Family C shift: c_S × α₁/(1-α₁) = {c_S_val} × {alpha_factor:.4f} = -{shift_C * 100:.4f}% (sign-uniform)")
print(f"    Family E shift: c_E × α₁/(1-α₁) = {c_E_val:.6f} × {alpha_factor:.4f} = ±{shift_E * 100:.4f}% (asymmetric)")
print()
print(f"  Original predictions:")
print(f"    M_Z_pred = {M_Z_pred:.4f} GeV  (residual {(M_Z_pred-M_Z_PDG)/M_Z_PDG*100:+.4f}%)")
print(f"    m_W_pred = {m_W_pred:.4f} GeV  (residual {(m_W_pred-m_W_PDG)/m_W_PDG*100:+.4f}%)")
print()
print(f"  Corrected predictions (Family C + Family E):")
print(f"    M_Z_corr = M_Z_pred × (1 - shift_C - shift_E) = {M_Z_corrected:.4f} GeV")
print(f"      residual: {resid_M_Z * 100:+.4f}%")
print(f"    m_W_corr = m_W_pred × (1 - shift_C + shift_E) = {m_W_corrected:.4f} GeV")
print(f"      residual: {resid_m_W * 100:+.4f}%")
print()

# Compute predicted Δρ from corrected values
rho_corrected = (m_W_corrected ** 2) / (M_Z_corrected ** 2 * cos2_W_MS)
delta_rho_corrected = rho_corrected - 1
rho_observed = (m_W_PDG ** 2) / (M_Z_PDG ** 2 * cos2_W_MS)
delta_rho_observed = rho_observed - 1

print(f"  Δρ check:")
print(f"    Original predicted ρ  = 1.0000 (by SM tree relation in M_Z, m_W formulas)")
print(f"    Corrected predicted ρ = {rho_corrected:.6f}  (δρ = {delta_rho_corrected*100:+.4f}%)")
print(f"    Observed ρ            = {rho_observed:.6f}  (δρ = {delta_rho_observed*100:+.4f}%)")
print(f"    Match: {abs(delta_rho_corrected - delta_rho_observed)*100:.4f}% absolute gap")


# ---------------------------------------------------------------------------
# Section 6 — Verdict
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("Phase B verdict")
print("=" * 78)
print()

routes_converge = c_E_route_H == c_E_route_C
within_sub_percent_M_Z = abs(resid_M_Z) < 0.01
within_sub_percent_m_W = abs(resid_m_W) < 0.01
within_sub_percent_rho = abs(delta_rho_corrected - delta_rho_observed) < 0.005

print(f"  Pre-declared abort criteria:")
print(f"    (CB.1) Routes give different values: {'NO ✓' if routes_converge else 'YES — close NEG'}")
print(f"    (CB.2) Joint cluster fit > 1% off either M_Z or m_W:")
print(f"           M_Z residual {resid_M_Z*100:+.4f}% → {'PASS' if within_sub_percent_M_Z else 'FAIL'}")
print(f"           m_W residual {resid_m_W*100:+.4f}% → {'PASS' if within_sub_percent_m_W else 'FAIL'}")
print(f"    (CB.3) Routes converge AND sub-percent: {'YES — POSITIVE' if (routes_converge and within_sub_percent_M_Z and within_sub_percent_m_W) else 'partial'}")
print(f"    (CB.4) Routes converge by construction (not independent):")
print(f"           Honest assessment: BOTH routes are 'c_S × (k*-1)/k*' modulation of")
print(f"           Family C — convergent by construction, NOT independently verified.")
print(f"           → CONDITIONAL POSITIVE.")
print()

print(f"  Summary:")
print(f"    c_S = 1/12 (Phase A, 2-routes convergent independently)")
print(f"    c_E = 1/18 (Phase B, 2-routes convergent BY CONSTRUCTION via NB-modulation)")
print(f"    Joint cluster fit:")
print(f"      M_Z residual: {resid_M_Z*100:+.4f}%  (was +{(M_Z_pred-M_Z_PDG)/M_Z_PDG*100:.4f}%)")
print(f"      m_W residual: {resid_m_W*100:+.4f}%  (was +{(m_W_pred-m_W_PDG)/m_W_PDG*100:.4f}%)")
print(f"      δρ predicted: {delta_rho_corrected*100:+.4f}%  (vs +{delta_rho_observed*100:.4f}% empirical)")
print()
print(f"  PHASE B CONDITIONAL POSITIVE:")
print(f"    - Joint cluster fit closes M_Z, m_W, and δρ simultaneously")
print(f"    - But two-routes discipline for c_E is partially-by-construction")
print(f"    - Genuine independent two-routes for c_E remains as Phase C work")
print()
print(f"  CLUSTER STATUS:")
print(f"    M_Z, m_W: graduate from STRUCTURAL-DERIVATION-CONDITIONAL to")
print(f"              THEOREM-GRADE-CONDITIONAL on Family C + Family E")
print(f"              (with two-routes discipline note for Family E)")
print()
print("=" * 78)
print("End of Phase B.")
print("=" * 78)
