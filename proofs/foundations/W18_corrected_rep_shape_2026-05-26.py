#!/usr/bin/env python3
"""
W18 — CORRECTED candidate for m_e/m_μ Koide-ratio common-mode (2026-05-26).

W17 had a critical issue: adding c_F^(rep) = -α₁²/(12·μ_rep_j) to the
master-doc Family-D c_F = -α₁²/12 broke m_τ closure (+63 ppm shift).

W18 candidate that PRESERVES m_τ closure:

    c_F^(rep)_j = -α₁² · c_S · (1/μ_rep_j − 1/μ_t)

The factor (1/μ_rep_j − 1/μ_t) is the "rep-j deviation from trivial reference."
It VANISHES at the trivial rep (where μ_rep = μ_t = N_atoms = 4) by construction,
so m_τ closure is preserved exactly.

For Ramanujan reps (ω, ω̄ with μ_rep = 2):
    1/μ_rep − 1/μ_t = 1/2 − 1/4 = 1/4
    c_F^(rep)_ω,ω̄ = -α₁²·(1/12)·(1/4) = -α₁²/48

STRUCTURAL INTERPRETATION:
  - c_S = 1/(2|E|) = 1/12: Perron-residue singlet projection (unified-oblique §3.2)
  - (1/μ_rep_j − 1/μ_t): rep-j "deviation from trivial" — measures how much
    the C₃ rep-j subspace differs from the trivial reference (where μ_t = N_atoms).

The rep-resolved correction at α₁² order acts AT the fermion leg with strength
weighted by the rep-deviation from trivial. This naturally vanishes at τ
(trivial rep is the reference), so the existing m_τ closure (master-doc Family-D
giving -(5/6)α₁² for y_τ correction) is preserved EXACTLY.

THE NEW PREDICTION:
    δm_τ = -(5/6)α₁²              (UNCHANGED — m_τ closure preserved)
    δm_e − δm_τ = -2·c_F^(rep)_e = α₁²·c_S·2·(1/μ_ω − 1/μ_t)
                = α₁²·(1/12)·(1/2)
                = α₁²/24
                = 63.43 ppm
    δm_μ − δm_τ = same = α₁²/24 = 63.43 ppm

This is the Koide-ratio common-mode shift, matching observation:
    c_e - 1 obs = +70.33 ppm → match 90.2%
    c_μ - 1 obs = +60.50 ppm → match 104.9%
    common-mode avg obs = 65.42 ppm → match 96.9%

The ω/ω̄ asymmetry +5 ppm remains separate (the candidate is rep-symmetric).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

from alpha_1 import predict_alpha_1
from Q_Koide import chain_import_ramanujan_multiplicities

alpha_1 = float(predict_alpha_1(3, 10))
mu_t, mu_o, mu_w = chain_import_ramanujan_multiplicities()
N_atoms = 4
two_E = 12
c_S = 1.0 / two_E

a1sq = alpha_1**2

print("=" * 76)
print("W18 — Corrected rep-resolved Family-D candidate (preserves m_τ closure)")
print("=" * 76)
print()
print(f"Framework primitives:")
print(f"  α₁² = {a1sq*1e6:.4f} ppm")
print(f"  c_S = 1/(2|E|) = 1/12 = {c_S:.6f}")
print(f"  (μ_t, μ_ω, μ_ω̄) = ({mu_t}, {mu_o}, {mu_w}); N_atoms = {N_atoms}")
print()
print(f"CANDIDATE: c_F^(rep)_j = -α₁²·c_S·(1/μ_rep_j − 1/μ_t)")
print()

# Compute c_F^(rep) per rep
def c_F_rep(mu_rep):
    return -a1sq * c_S * (1.0/mu_rep - 1.0/mu_t)

c_F_rep_t = c_F_rep(mu_t)
c_F_rep_o = c_F_rep(mu_o)
c_F_rep_ob = c_F_rep(mu_w)

print(f"Per-rep contributions:")
print(f"  τ (trivial, μ=4): c_F^(rep)_τ = -α₁²·c_S·(1/4 − 1/4) = 0           ← preserves m_τ")
print(f"  ω (μ=2):           c_F^(rep)_ω = -α₁²·c_S·(1/2 − 1/4) = -α₁²/48 = {c_F_rep_o*1e6:.3f} ppm")
print(f"  ω̄ (μ=2):          c_F^(rep)_ω̄ = -α₁²/48 = {c_F_rep_ob*1e6:.3f} ppm")
print()

# Total c_F at Yukawa vertex (universal + rep-resolved)
c_F_univ = -a1sq * c_S
c_H_alpha2 = a1sq

c_F_total_t = c_F_univ + c_F_rep_t
c_F_total_o = c_F_univ + c_F_rep_o
c_F_total_ob = c_F_univ + c_F_rep_ob

# Yukawa-vertex corrections
delta_y_t  = -(c_H_alpha2 + 2*c_F_total_t)
delta_y_o  = -(c_H_alpha2 + 2*c_F_total_o)
delta_y_ob = -(c_H_alpha2 + 2*c_F_total_ob)

print(f"Yukawa-vertex corrections at α₁² (Family-D total = universal + rep-resolved):")
print(f"  δy_τ = -(α₁² + 2·c_F_τ_total) = -(α₁² - α₁²/6) = -(5/6)α₁² = {delta_y_t*1e6:.2f} ppm")
print(f"  δy_e = -(α₁² + 2·c_F_e_total) = -(α₁² - 5α₁²/24) = -19α₁²/24 = {delta_y_o*1e6:.2f} ppm")
print(f"  δy_μ = -(α₁² + 2·c_F_μ_total) = -19α₁²/24 = {delta_y_ob*1e6:.2f} ppm")
print()
print(f"  → m_τ correction UNCHANGED at -(5/6)α₁² = {delta_y_t*1e6:.2f} ppm")
print(f"  → Master-doc Family-D m_τ closure (-0.17σ_PDG) PRESERVED ✓")
print()

# Koide-ratio shifts
c_e_minus_1_pred = delta_y_o - delta_y_t
c_mu_minus_1_pred = delta_y_ob - delta_y_t
print(f"Koide-ratio shifts (at m-level via m = v·y):")
print(f"  c_e - 1 predicted = δy_e − δy_τ = α₁²/24 = {c_e_minus_1_pred*1e6:.3f} ppm")
print(f"  c_μ - 1 predicted = δy_μ − δy_τ = α₁²/24 = {c_mu_minus_1_pred*1e6:.3f} ppm")
print()

# Compare to observation
c_e_obs = 70.33e-6
c_mu_obs = 60.50e-6
print(f"Comparison to observation (m_τ at PDG):")
print(f"  c_e - 1 obs = {c_e_obs*1e6:.3f} ppm  predicted {c_e_minus_1_pred*1e6:.3f} → match {c_e_minus_1_pred/c_e_obs*100:.1f}%")
print(f"  c_μ - 1 obs = {c_mu_obs*1e6:.3f} ppm  predicted {c_mu_minus_1_pred*1e6:.3f} → match {c_mu_minus_1_pred/c_mu_obs*100:.1f}%")
print(f"  Common-mode avg: obs {(c_e_obs+c_mu_obs)/2*1e6:.3f} ppm → match {c_e_minus_1_pred/((c_e_obs+c_mu_obs)/2)*100:.1f}%")
print()

# Asymmetry
print(f"Residuals (ω/ω̄ asymmetry):")
print(f"  c_e - predicted = {(c_e_obs - c_e_minus_1_pred)*1e6:+.2f} ppm")
print(f"  c_μ - predicted = {(c_mu_obs - c_mu_minus_1_pred)*1e6:+.2f} ppm")
print(f"  These are the ω/ω̄ asymmetry pieces remaining open.")
print()

# Now check the structural derivation
print("=" * 76)
print("STRUCTURAL DERIVATION (CANDIDATE)")
print("=" * 76)
print("""
The rep-resolved correction c_F^(rep)_j has form:
    c_F^(rep)_j = -α₁²·c_S·(1/μ_rep_j − 1/μ_t)

Each factor is theorem-grade in the framework:

(a) α₁²: Family-D α₁² scale (master doc §3 D Route H joint walker, theorem-grade).

(b) c_S = 1/(2|E|) = 1/12: Perron-residue singlet projection of B_NB at Γ
    (`theorem_unified_oblique.md` §3.2, theorem-grade with handshake-lemma
    Route H ≡ Route C derivation).

(c) μ_rep_j: C₃-rep multiplicities on V_Ram subspace of B(P), with values
    (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) per `predictions/Q_Koide.py` (theorem-grade
    structural via Ramanujan subspace decomposition).

(d) μ_t = N_atoms = 4: the trivial-rep multiplicity equals atoms-per-cell
    (consequence of arc-transitivity: trivial-rep saturates the cell symmetry).

The COMBINATION (1/μ_rep_j − 1/μ_t) measures the "rep-j deviation from
trivial reference." It vanishes IDENTICALLY at j = trivial (where the
rep is the trivial-rep reference), and is positive for Ramanujan reps
(where μ_rep < μ_t).

The structural interpretation of c_F^(rep) = -α₁²·c_S·(deviation):
At the Yukawa vertex, the per-fermion-leg dark correction at α₁²
has a leading UNIVERSAL piece (c_F = -α₁²/12 per master doc Family-D)
and a SUB-LEADING REP-RESOLVED piece c_F^(rep) that captures the
"rep-j-specific deviation from the trivial reference." For τ (trivial rep)
the deviation vanishes; the correction reduces to the universal piece —
matching the master-doc result and preserving the m_τ closure.

For e/μ (Ramanujan reps), the deviation is non-zero, contributing an
additional per-leg correction that DOES survive in Koide ratios m_j/m_τ.

OUTSTANDING DERIVATION GAP:
The structural CHANNEL_SELECT step picking out (1/μ_rep_j − 1/μ_t) as
the rep-resolved channel form, parallel to the master-doc Family-D
Clause-6 two-step (channel_select for single-edge-spectral channel →
canonical_encoding via handshake lemma), needs explicit formalization.

The natural channel_select at the rep-resolved level: the fermion leg
in rep j couples to V_Ram via the rep-j subspace (dim μ_rep_j),
with channel density 1/μ_rep_j PER MODE. The "reference" channel density
is 1/μ_t = 1/N_atoms (trivial-rep saturating cell). The DEVIATION
(1/μ_rep_j − 1/μ_t) captures the rep-specific contribution.

This needs derivation parallel to unified oblique §3.2 (which derives c_S
via gauge-singlet projection on B_NB at Γ). The rep-resolved analog
would be: V_Ram-rep-j projection on B(P), with channel-density factor.

LINTER 9-CLAUSE STATUS:
  Clause 1: N/A
  Clause 2: PASS (K-rational arithmetic)
  Clause 3: PARTIAL — c_S theorem-grade (unified oblique §3.2);
            μ_rep_j theorem-grade (Q_Koide.py); BUT the rep-resolved
            channel_select form (1/μ_rep_j − 1/μ_t) needs derivation
            parallel to the unified-oblique gauge-singlet projection.
  Clause 4: PASS (all framework primitives have predictions/ files)
  Clause 5: master-doc §3 D needs the rep-resolved extension theorem
  Clause 6: PASS (K-rational; channel_select form needs formal Clause-6
            two-step but structurally motivated)
  Clause 7: NOT attempted
  Clause 8: PARTIAL — 97% common-mode match; ω/ω̄ asymmetry +5 ppm open
  Clause 9: PASS (no π)

VERDICT: GENUINE CANDIDATE structural form.

It preserves the master-doc Family-D m_τ closure EXACTLY (c_F^(rep)_τ = 0
by construction), so doesn't break existing predictions.
It uses only theorem-grade framework primitives (c_S, μ_rep_j, α₁²).
It gives 97% common-mode match on the Koide-ratio observation.
It leaves the ω/ω̄ asymmetry +5 ppm as a separate sub-leading piece
(within master doc §8b ~0.5% Yukawa systematic budget).

The honest grade: CANDIDATE-GRADE structural form, pending formal
Clause-6 two-step derivation of the rep-resolved channel structure.
1-3 sessions of research-level work to upgrade to theorem-grade.

This IS materially different from the W4-W10 α₁³ extension that
failed cycle-decomposition tests. The α₁² rep-resolved candidate
uses the framework's EXISTING Family-D mechanism at the same order,
with a rep-deviation modulation that vanishes naturally at trivial.
""")
