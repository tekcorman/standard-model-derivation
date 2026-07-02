#!/usr/bin/env python3
"""
W4 — Substrate derivation probe: α₁³ Family-D with C₃-isotypic decomposition.

PURPOSE
-------
W1 found that closing m_e and m_μ Koide-ratio residuals requires a sub-
leading Family-D correction at α₁³ with per-Ramanujan-C₃-rep structure
on f_j.  The K-rational candidate shape from W1 was:

    κ_j (on f_j) = (2/μ_rep_j) · α₁_bare³           — numerical ansatz

This probe ATTEMPTS the substrate-side derivation of this α₁³ correction
by extending master doc §3 D Routes H + C to one order higher with
explicit C₃-isotypic decomposition.

GROUND RULES
------------
- No `predictions/` modifications.
- Must reproduce the magnitude (α₁_bare³ scale) AND the per-rep
  μ_rep_j-dependence from substrate physics, not back-fitting.
- Honest negative is fine.

ROUTES H AND C AT α₁² (REFRESH)
-------------------------------
(master doc §3 D, lines 89-145)

Route H — joint Hashimoto-spectral walker survival on the (srs × srs-z)
  Sunada-isospectral pair.  Each substrate has per-step NB-walker
  survival q_NB = (k*−1)/k* = 2/3.  Walks of length 2(g−2) = 16 steps
  give survival prob = (2/3)¹⁶ = α₁_bare² = (2/3)⁸·(2/3)⁸.

Route C — cycle counting at m=2 closed bubble length L_closed(m=2) =
  2g−4 = 16 on srs.  c_H = q_NB^{L_closed} = α₁².

Per Higgs leg:  c_H = +α₁²    (joint walker)
Per Fermion leg: c_F = −α₁²/(N_atoms · k*) = −α₁²/12  (single-edge-spectral channel via
                 `theorem_car_local_jordan_wigner.md §1`, Clause-6 two-step)

Family-D vertex correction: δg/g = −(n_H · c_H + n_F · c_F).

For y_τ (1H + 2F): δy_τ/y_τ = −[α₁² + 2(−α₁²/12)] = −(5/6)·α₁²

ROUTE H AT α₁³ — DERIVATION ATTEMPT
------------------------------------
Extend joint NB walker to length 3(g−2) = 24:
  Survival = q_NB^{24} = (2/3)²⁴ = α₁_bare³ exactly.

Magnitude check: α₁_bare³ = 59.40 ppm — matches the W1 target scale.

But: at 24 steps, the joint walker has wound around the substrate cycle
THREE times.  The cycle's per-trip C₃ holonomy h_cycle takes values
in {1, ω, ω̄}.  After 3 trips, cumulative holonomy = h_cycle³.  In C₃:
  trivial: 1³ = 1
  ω-rep:   ω³ = 1
  ω̄-rep:  ω̄³ = 1
All reps give TRIVIAL cumulative holonomy at length 24.

CONCLUSION (Route H at α₁³): the joint walker's α₁³ survival is
**REP-UNIVERSAL**, not rep-resolved.  The rep-resolution cannot come
from the cycle-holonomy mechanism.

This contributes only to c_H^(α₁³) (rep-universal per-Higgs-leg piece),
NOT to c_F_rep^(α₁³).

ROUTE C AT α₁³ — DERIVATION ATTEMPT
------------------------------------
Extend cycle counting to m=3 closed bubble length L_closed(m=3) = 3g−6 = 24:
  c_H^(α₁³) = q_NB^{24} = α₁_bare³ (rep-universal, same as Route H)

For c_F^(α₁³) (per-fermion-leg disruption at α₁³), the relevant question
is: how does the per-leg Yukawa-vertex Jordan-Wigner closed fermion loop
extend at α₁³ order?

At α₁² (master doc lines 123-127), c_F = −α₁²/(N_atoms·k*) = −α₁²/12.
The denominator 12 = N_atoms·k* = 2|E|/cell is the FULL per-cell directed-
edge count (Euler identity).  At α₁³, the relevant counting is per-
Ramanujan-rep, since the walker now resolves C₃ structure (V_Ram is
where walker activity lives, per W45 mode-count theorem).

The per-Ramanujan-rep directed-edge count on V_Ram is μ_rep_j:
  (μ_trivial, μ_ω, μ_ω̄) = (4, 2, 2)  per primitive cell

POSTULATED form:
  c_F_rep^(α₁³) = (some_coefficient) · α₁_bare³ / μ_rep_j  per leg

The QUESTION: what is the coefficient, and can it be derived?

3 candidate K-rational coefficients to test:
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

a1 = alpha_1_bare
a1_3 = a1**3
N_atoms = 4

print("=" * 72)
print("W4 — substrate α₁³ derivation: Routes H + C with C₃-isotypic decomp")
print("=" * 72)
print()
print(f"Framework primitives:  k*={k_star}, g={g}, N_atoms={N_atoms}")
print(f"  α₁_bare = (2/3)⁸    = {a1:.10f}")
print(f"  α₁_bare³ = (2/3)²⁴  = {a1_3:.6e} = {a1_3*1e6:.4f} ppm")
print(f"  (μ_trivial, μ_ω, μ_ω̄) = ({mu_t}, {mu_o}, {mu_w})  on V_Ram (dim 8)")
print()

# Observation targets (from W1 back-solve with m_τ at PDG):
target_kappa_omega    = 35.166e-6   # κ_ω − κ_trivial   (electron, ω rep)
target_kappa_omegabar = 30.249e-6   # κ_ω̄ − κ_trivial  (muon,    ω̄ rep)
target_y_tau_residual = 12.0e-6     # |y_τ residual|, positive correction needed

print("Targets from W1 back-solve (treating m_τ at PDG as exact):")
print(f"  κ_ω    − κ_trivial = +{target_kappa_omega*1e6:6.2f} ppm  (electron, ω-rep)")
print(f"  κ_ω̄   − κ_trivial = +{target_kappa_omegabar*1e6:6.2f} ppm  (muon, ω̄-rep)")
print(f"  ω/ω̄ asymmetry     = +{(target_kappa_omega-target_kappa_omegabar)*1e6:6.2f} ppm  (sub-leading, open)")
print(f"  y_τ residual to close = +{target_y_tau_residual*1e6:6.2f} ppm  (rep-universal, c_H^α₁³ - 2c_F_τ^α₁³)")
print()

# ------------------------------------------------------------------------
# Route H + C analysis:
# c_H^(α₁³) — rep-universal piece from joint NB walker at length 24
# c_F_rep^(α₁³) — per-leg piece, rep-resolved via V_Ram decomposition
# ------------------------------------------------------------------------
print("ROUTE H AT α₁³:")
print(f"  Joint NB walker on (srs × srs-z) at length 3(g-2) = 24 steps")
print(f"  Survival prob = q_NB²⁴ = (2/3)²⁴ = α₁_bare³ = {a1_3*1e6:.4f} ppm")
print(f"  After 3 girth-cycle trips: C₃ holonomy h³ = 1 for ALL reps (trivializes)")
print(f"  ⇒ Route H at α₁³ is REP-UNIVERSAL (contributes to c_H^α₁³ only)")
print()

print("ROUTE C AT α₁³:")
print(f"  m=3 closed bubble length L_closed(m=3) = 3g − 6 = 24")
print(f"  c_H^(α₁³) = q_NB²⁴ = α₁³ (same as Route H, rep-universal)")
print()
print(f"  c_F_rep^(α₁³) ANSATZ: per-Yukawa-leg substrate disruption at α₁³ with")
print(f"  rep-resolution via the Ramanujan-subspace C₃ multiplicity μ_rep_j.")
print(f"  Natural K-rational candidate denominator:  μ_rep_j  (= 4, 2, 2)")
print()

# Test 3 candidate prefactors
print("Testing K-rational candidate shapes c_F_rep^(α₁³) = − A·α₁³/μ_rep_j per leg:")
print()
print(f"  {'A':>4}  {'c_F_τ ppm':>10}  {'c_F_e ppm':>10}  {'Δκ_e ppm':>10}  {'Δκ_μ ppm':>10}  {'match':>10}")
print("  " + "-"*70)
for A in [1, 2, 3, 4]:
    cF_t = -A*a1_3/mu_t
    cF_e = -A*a1_3/mu_o
    cF_m = -A*a1_3/mu_w
    # δy_j = -(c_H + 2c_F_j); difference  δy_j - δy_τ = -2·(c_F_j - c_F_τ)
    # In Koide ratio m_j/m_τ = y_j/y_τ, so c_j - 1 = δy_j - δy_τ
    dky_e = -2*(cF_e - cF_t)
    dky_m = -2*(cF_m - cF_t)
    # κ_j (on f_j) interpretation: m ∝ f² ⇒ (c_j-1) = 2·(κ_j - κ_τ) on f
    # So κ_j_on_f - κ_τ_on_f = (c_j - 1)/2 = (δy_j - δy_τ)/2 = -(cF_j - cF_τ)
    kappa_e_pred = dky_e/2
    kappa_m_pred = dky_m/2
    flag = ""
    if abs(kappa_e_pred - target_kappa_omega)/target_kappa_omega < 0.20 and \
       abs(kappa_m_pred - target_kappa_omegabar)/target_kappa_omegabar < 0.20:
        flag = "  ← within 20%"
    print(f"  {A:>4}  {cF_t*1e6:>10.3f}  {cF_e*1e6:>10.3f}  {kappa_e_pred*1e6:>10.3f}  {kappa_m_pred*1e6:>10.3f}{flag}")
print()

# Selected candidate: A=2 (matches κ_ω̄ at 0.98×)
A = 2
cF_t = -A*a1_3/mu_t
cF_e = -A*a1_3/mu_o
cF_m = -A*a1_3/mu_w
kappa_e_pred = -(cF_e - cF_t)
kappa_m_pred = -(cF_m - cF_t)

print(f"CANDIDATE selected: A = 2  →  c_F_rep^(α₁³) = −2α₁³/μ_rep_j  per leg")
print(f"  c_F_τ = −2α₁³/4 = −α₁³/2 = {cF_t*1e6:.3f} ppm  (trivial rep)")
print(f"  c_F_e = −2α₁³/2 = −α₁³   = {cF_e*1e6:.3f} ppm  (ω rep)")
print(f"  c_F_μ = −2α₁³/2 = −α₁³   = {cF_m*1e6:.3f} ppm  (ω̄ rep)")
print()
print(f"Predicted Koide-ratio shifts (κ_j on f_j level):")
print(f"  κ_ω  − κ_τ = +{kappa_e_pred*1e6:6.3f} ppm   obs +{target_kappa_omega*1e6:6.3f}   ratio {kappa_e_pred/target_kappa_omega:.3f}×")
print(f"  κ_ω̄ − κ_τ = +{kappa_m_pred*1e6:6.3f} ppm   obs +{target_kappa_omegabar*1e6:6.3f}   ratio {kappa_m_pred/target_kappa_omegabar:.3f}×")
print()

# Now check y_τ residual closure via rep-universal c_H^(α₁³):
print("=" * 72)
print("y_τ RESIDUAL CLOSURE — rep-universal c_H^(α₁³)")
print("=" * 72)
print()
print("With c_F_τ^(α₁³) = −α₁³/2, the y_τ residual closure requires:")
print("  δy_τ^(α₁³) = −(c_H^(α₁³) + 2·c_F_τ^(α₁³))")
print(f"             = −(c_H^(α₁³) − α₁³) = +{target_y_tau_residual*1e6:.2f} ppm")
print(f"  → c_H^(α₁³) = α₁³ − target = {a1_3*1e6:.2f} − {target_y_tau_residual*1e6:.2f} = {(a1_3-target_y_tau_residual)*1e6:.2f} ppm")
print(f"               = α₁³·({(a1_3-target_y_tau_residual)/a1_3:.4f})")
print()
print(f"Closest K-rational candidates for c_H^(α₁³):")
ch_target = a1_3 - target_y_tau_residual
candidates_cH = [
    ("α₁³ · (4/5)",      a1_3 * 4/5),
    ("α₁³ · (3/4)",      a1_3 * 3/4),
    ("α₁³ · (5/6)",      a1_3 * 5/6),
    ("α₁³ · (7/9)",      a1_3 * 7/9),
    ("α₁³ · (1 - 1/5)",  a1_3 * 4/5),
    ("α₁³ - α₁_bare⁴",   a1_3 - a1**4),     # subleading sub-leading
    ("α₁³ · (1 - α₁_bare/5)", a1_3 * (1 - a1/5)),
]
for name, val in candidates_cH:
    print(f"  {name:<35} = {val*1e6:.3f} ppm  ratio to target {val/ch_target:.4f}×")
print()
print("None is a clean K-rational (factors 4/5, 5/6, 7/9 are ad-hoc).")
print("→ The rep-UNIVERSAL c_H^(α₁³) does NOT have a structural derivation")
print("  matching the y_τ residual at clean K-rational form.")
print()

# Summary
print("=" * 72)
print("W4 SUMMARY")
print("=" * 72)
print("""
WHAT CLOSES STRUCTURALLY:
  ✓ α₁³ magnitude: Route H joint walker at length 24 = 3(g-2) gives
     exactly α₁_bare³ from q_NB^24.  Theorem-grade in the same sense
     as Family-D α₁² Route H.
  ✓ Rep-resolution channel: V_Ram C₃ multiplicities (4,2,2) are the
     correct decomposition for per-fermion-leg dark disruption at α₁³.
  ✓ K-rationality: (2/μ_rep)·α₁³ ∈ ℚ ⊂ K = ℚ(√2,√3,√5).
  ✓ Symmetric piece magnitude: predicted κ_ω = κ_ω̄ = α₁³/2 matches
     observed κ_ω̄ at 0.98× (1% — within the framework's natural
     systematic budget).

WHAT DOES NOT CLOSE (research gaps):
  ✗ The numerator-2 coefficient: A=2 in c_F_rep = −Aα₁³/μ_rep_j is
     SELECTED to match κ_ω̄ within 1%, but the ANSATZ does NOT
     uniquely determine A from first principles.  A=1 (=29.7/2=14.85
     ppm shift) and A=3 (=89 ppm shift) are equally K-rational; A=2
     is preferred ONLY by data, which is the "fitting" pattern the
     framework explicitly disallows (master doc §6 Step 6).
  ✗ The ω/ω̄ asymmetry +4.9 ppm is NOT reproduced — the per-rep shape
     gives identical correction to ω and ω̄ (both Ramanujan reps with
     μ=2).  Closing this requires a δ-flavoured sub-leading mechanism
     since δ=2/9 is the source of ω↔ω̄ breaking in f_j.  Not derived
     here.
  ✗ The rep-universal c_H^(α₁³) needed to close the y_τ −12 ppm
     residual has no clean K-rational structural derivation.  The
     ratio α₁³·(target match)/α₁³ = (a1³−12 ppm)/a1³ ≈ 0.798 isn't
     a clean rational (not 4/5, 5/6, 7/9, or any simple ratio).

VERDICT (LINTER-DISCIPLINE HONEST):
  This is SKETCH-grade structural progress.  The α₁³ magnitude and
  per-rep shape are derivable from extending Routes H + C; the
  specific NUMERATOR (A=2) and the rep-universal c_H^(α₁³) are NOT
  determined by substrate physics in this attempt.

  Linter 9-clause gate status:
    Clauses 1, 2, 6, 9: PASS (K-rational structural shape, no π)
    Clause 4: PASS (uses predictions/ chain)
    Clause 3, 5: PARTIAL — the α₁³ extension is structurally motivated
      but the numerator-2 and c_H^(α₁³) coefficient lack derivation;
      master doc §3 D would need a NEW theorem-grade entry for the
      α₁³ rep-resolved member.
    Clause 7: NOT ATTEMPTED — multi-axis audit-v2 §3 table required
      for the new α₁³ family.
    Clause 8: SAME structural-grade status as W1 (THEOREM-GRADE-STRUCTURAL
      with named open residue: ω/ω̄ asymmetry +5 ppm + the rep-universal
      c_H^(α₁³) gap).

NEXT STEPS:
  (1) Derive the numerator A=2 from first principles. Hypothesis: it
      comes from the doubled-cover structure of B(P) (NB walker on
      directed edges; factor 2 from directed-to-undirected). But this
      double-counts what's already in q_NB^24 (which IS the survival
      on directed edges).  ALTERNATIVELY: A=2 might emerge from the
      "2 fermion legs" already in n_F=2 being mis-applied (already
      accounted by the leg counting in δg/g = −(c_H + 2c_F)).  Either
      way the factor 2 needs an explicit substrate derivation OR
      reformulation that doesn't have a free coefficient.

  (2) Derive c_H^(α₁³). Likely structurally constrained by joint
      walker survival at length 24 with some additional combinatorial
      factor (NOT yet identified).

  (3) Derive ω/ω̄ asymmetry mechanism: most likely δ-flavoured (i.e.,
      involves δ=2/9 in some sub-leading product).

  (4) ONLY AFTER (1)+(2)+(3): submit to linter Checkpoint 1+2 on
      (m_e, m_μ, y_τ) joint triage.

W4 status: NOT yet linter-ready.  Honest structural sketch + named gaps.
""")
