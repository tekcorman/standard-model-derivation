#!/usr/bin/env python3
"""
W3 — Survey: where else should the per-C₃-rep correction apply (2026-05-26)?

PURPOSE
-------
W1 conjectured a per-C₃-rep Family-D α₁³ correction:
    κ_j = 2·α₁_bare³ / μ_rep_j     (per-f_j multiplicative)
This shape uses ONLY (a) α₁_bare (substrate walk statistics) and
(b) μ_rep_j (the C₃ multiplicity of generation j's rep on V_Ram).

W3 maps where this shape OUGHT to apply in the framework's prediction set,
and where it should NOT.  This isn't a fix — it's a relevance map that
informs (i) whether W1's substrate-derivation push is high-value and
(ii) which downstream predictions would be affected if W1 closes.

Per `docs/parameters/parameter_linter.md`, this is `proofs/` exploratory
work — NO `predictions/` modifications.

WHERE THE SHAPE OUGHT TO APPLY
------------------------------
The per-C₃-rep correction applies wherever a prediction:
  (1) involves a Yukawa-like vertex with fermion legs assigned to
      DIFFERENT C₃ reps within a single observable; AND
  (2) the Family-D leading α₁² piece is rep-INDEPENDENT (so cancels in
      ratios) leaving an α₁³ rep-DEPENDENT residue.

This pinpoints OBSERVABLES BUILT FROM RATIOS WITHIN A C₃ GENERATION
TRIPLET:

  CHARGED LEPTONS (Koide triplet)
    m_e:    f_min (ω rep)       — W1 case
    m_μ:    f_mid (ω̄ rep)      — W1 case
    m_τ:    f_max (trivial rep) — sets the reference

  DOWN QUARKS (Koide-down triplet, ε²_down ≈ 2.388)
    m_d:    ω rep  — affected if Koide structure applies
    m_s:    ω̄ rep — affected
    m_b:    trivial — reference

  UP QUARKS (Koide-up triplet, ε²_up ≈ 3.094)
    m_u:    ω rep — affected
    m_c:    ω̄ rep — affected
    m_t:    trivial — reference

BUT: charged-quark Koide ratios are NOT currently predicted — light quark
masses are blocked on Need-D-3 per `[[reference-quark-mass-entry-point-
2026-05-25]]` and m_t is retracted.  So the W1 mechanism applies in
principle but has no live downstream prediction to affect.

PMNS angles (θ_12, θ_13, θ_23)
    These use the FULL Ramanujan amplitude structure, not single-rep
    ratios.  They mix across reps rather than pick one — so per-rep
    correction would enter as a tensor on the mixing matrix, NOT as
    multiplicative on the angle.  Different mechanism family.

NEUTRINO MASSES (m_ν₂, m_ν₃)
    m_ν₃ is set by global spectral gap (k*·N_atoms)·M_Pl·N_hub^{-1/2};
    no C₃-rep assignment in the formula.
    m_ν₂ is via R = 228/7 splitting (Ihara spectral, K_4 topology) —
    not a Koide-rep ratio.
    → Per-rep correction does NOT apply.

WHERE THE SHAPE EXPLICITLY DOES NOT APPLY
-----------------------------------------
  • V_us, V_cb, V_ub — full G_NB-Bloch amplitudes, no single-rep host
  • J_CKM, δ_CP — geometric / over-determined via V_{−1}-T_{B-L}
  • δ_r, δρ — oblique self-energies, no Yukawa vertex
  • y_τ, y_b, y_t — bare Yukawa values are per-generation tree-level
    couplings; the per-rep correction enters Family-D's α₁³ piece on
    the ABSOLUTE Yukawa (not the inter-generation ratio).  This is the
    W2-companion piece (m_τ −13 ppm common-mode); SAME mechanism family
    but rep-UNIVERSAL slice, not rep-DEPENDENT.
  • Gauge couplings, α_GUT, M_Z, m_W, m_H, v, M_unif — none use Koide
    f_j or C₃-rep assignment on V_Ram.

SO THE PER-REP SHAPE IS RELEVANT FOR:
  • m_e, m_μ (live; W1 target)             ~30 ppm scale
  • m_u, m_d, m_s, m_c (not yet predicted)  ~?? scale (Need-D-3 wall)
  • PMNS-mixing TENSOR (different formal structure)

THE FRAMEWORK-NATIVE COUNTERPART CHECK (12-OBSERVABLE §8 FAMILY)
----------------------------------------------------------------
Memory: 12 observables already over-determined at theorem-grade
(y_t, y_b, V_us, V_cb, V_ub, δ_r, δρ, y_τ, θ_12, θ_13, θ_23, A_s).
None of these is a Koide-triplet RATIO within one generation set.
Charged-lepton mass ratios (m_e/m_τ, m_μ/m_τ) are NOT in the §8 family
— precisely because the Koide-ratio Clause-6 channel_select is a
DIFFERENT mechanism than the §8 amplitude/charge-counting machinery.

PREDICTION (and falsifier)
--------------------------
If W1's substrate derivation closes, the framework will gain:
  - tighter m_e, m_μ predictions (~10 ppm residual, was ~80 ppm)
  - NO change to the 12-observable §8 family (different mechanism)
  - NO change to neutrino masses or PMNS angles (different mechanism)
  - LATENT change to light quark masses IF Need-D-3 closes —
    those would inherit the same (2/μ_rep)·α₁³ correction structure

Falsifier: if Need-D-3 closes and the resulting quark Koide ratios
DON'T show similar per-rep residuals at the (2/μ_rep)·α₁³ magnitude,
W1's per-rep mechanism is falsified (or rep-resolved only for
leptons — which would itself be a structural finding).

W3 → no predictions changed.  Bounded relevance map recorded.
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from Q_Koide import chain_import_ramanujan_multiplicities

d = predict_d_spatial()
k_star = int(round(predict_k_star(d)))
g = predict_g_girth(k_star, d)
alpha_1_bare = float(predict_alpha_1(k_star, g))
mu_t, mu_o, mu_w = chain_import_ramanujan_multiplicities()

print("=" * 70)
print("W3 — Per-C₃-rep mechanism applicability survey (2026-05-26)")
print("=" * 70)
print()
print("Shape: κ_j = 2·α₁_bare³ / μ_rep_j  on f_j  (per-Yukawa-vertex,")
print(f"per-Ramanujan-C₃-rep). α₁_bare³ = {alpha_1_bare**3*1e6:.2f} ppm.")
print()

categories = [
    # (name, applies?, mechanism, scale, status)
    ("CHARGED LEPTONS — m_e, m_μ via Koide",          True,  "per-rep f_j correction",            "~30 ppm",   "LIVE — W1 target"),
    ("CHARGED LEPTONS — m_τ absolute scale",          False, "rep-universal α₁³ Family-D",        "~12 ppm",   "W2 piece (sub-leading Feshbach)"),
    ("DOWN QUARKS — m_d, m_s, m_b via Koide-down",    True,  "same per-rep mechanism (if Need-D-3 closes)", "~?? ppm",   "BLOCKED on Need-D-3"),
    ("UP QUARKS — m_u, m_c, m_t via Koide-up",        True,  "same per-rep mechanism (if Need-D-3 closes)", "~?? ppm",   "BLOCKED on Need-D-3 + m_t retracted"),
    ("NEUTRINO MASSES — m_ν₂, m_ν₃",                  False, "global spectral gap; R = 228/7 Ihara", "n/a",       "no C₃-rep assignment in formula"),
    ("PMNS ANGLES — θ_12, θ_13, θ_23",                False, "FULL Ramanujan amplitude (not single-rep ratio)", "n/a", "different mechanism family (mix-tensor)"),
    ("CKM AMPLITUDES — V_us, V_cb, V_ub",             False, "G_NB Bloch-integrated, not single fiber", "n/a",    "different mechanism family"),
    ("OBLIQUE — δ_r, δρ",                             False, "self-energies, no Yukawa vertex",   "n/a",       "different mechanism family"),
    ("Y_τ, Y_b, Y_t (absolute Yukawas)",              True,  "rep-universal α₁³ Family-D piece",  "~12-? ppm", "W2 sibling; rep-UNIVERSAL not rep-DEPENDENT"),
    ("GAUGE COUPLINGS — α_GUT, g_1/2/3, sin²θ_W",     False, "no Yukawa vertex",                  "n/a",       "different mechanism family"),
    ("Higgs sector — m_H, λ, v",                      False, "no C₃-rep assignment on V_Ram",     "n/a",       "different mechanism family"),
]

print(f"  {'observable':<46} {'applies':>8}  {'mechanism':<48} {'scale':>10}  status")
print(f"  {'-'*46} {'-'*7}  {'-'*48} {'-'*10}  ------")
for name, applies, mechanism, scale, status in categories:
    marker = "  YES" if applies else "  no "
    print(f"  {name:<46} {marker:>8}  {mechanism:<48} {scale:>10}  {status}")
print()

print("=" * 70)
print("Family-D mechanism decomposition (per W1+W2)")
print("=" * 70)
print()
print("Family-D vertex correction (master doc §3 D):")
print("  Leading: y_j = y_j_tree · (1 - (5/6)·α₁²)        rep-universal, cancels in ratios")
print("  α₁³ rep-UNIVERSAL:  +(c_univ)·α₁³                NOT yet derived — drives m_τ −13 ppm (W2)")
print("  α₁³ rep-DEPENDENT:  +(2/μ_rep_j)·α₁³ on f_j      NOT yet derived — drives Koide ratios (W1)")
print()
print("Both α₁³ pieces are RESEARCH-LEVEL within the same Family-D extension.")
print("Closing them is a single coupled work item, not two independent ones.")
print()
print("=" * 70)
print("VALIDATION CHECK against the 12-observable §8 family")
print("=" * 70)
print()
print("Memory says 12 observables are over-determined at theorem-grade:")
print("  Quark/Gauge:  y_t, y_b, V_us, V_cb, V_ub, δ_r, δρ           (7)")
print("  Lepton/PMNS:  y_τ, θ_12, θ_13, θ_23                          (4)")
print("  Cosmology:    A_s                                              (1)")
print()
print("NONE of these is a Koide-triplet RATIO within one generation set.")
print("So the per-rep mechanism would NOT show up in the §8 family —")
print("consistent with the §8 over-determination remaining intact even if")
print("W1 closes.  This is a falsifier-pass: the per-rep mechanism does")
print("NOT conflict with the framework's existing theorem-grade catalogue.")
print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()
print("Closing W1's substrate derivation (α₁³ rep-resolved Family-D) is")
print("HIGH-VALUE because:")
print("  • It closes m_e, m_μ to ~10 ppm (W1 estimate).")
print("  • It SIMULTANEOUSLY closes m_τ −13 ppm via the rep-universal sibling (W2).")
print("  • It would automatically apply to light quark Koide ratios IF Need-D-3 closes,")
print("    AT NO ADDITIONAL DERIVATION COST.  Same α₁³ mechanism, same K-rational shape.")
print("  • The §8 family remains untouched (no conflict).")
print()
print("The bounded probe target is therefore:")
print("  PROBE: Compute the next-order Family-D correction by extending")
print("  Route H or Route C (master doc §3 D) to α₁³, with explicit")
print("  C₃-isotypic decomposition of the 3-step substrate walk.")
print("  Confirm BOTH (a) rep-universal piece reproduces y_τ −12 ppm")
print("  with correct sign, AND (b) rep-dependent piece reproduces the")
print("  C₃-symmetric (2/μ_rep)·α₁³ shape.  The ω/ω̄ asymmetry +5 ppm is")
print("  a separate sub-leading mechanism (δ-flavoured).")
print()
print("W3 → no predictions changed.")
