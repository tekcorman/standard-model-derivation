#!/usr/bin/env python3
"""
proofs/foundations/substrate_Delta_alpha_photon_channel_2026-05-16.py

ATTEMPT: substrate Δα as the PHOTON Π_γγ channel of the one B_NB(srs)
resolvent — same method as δ_r (Perron residue) / S (tree-cover neutral
Γ→P flow).  Run under parameter_linter.md discipline: Checkpoint-1 grep
done (see header notes); the HARD QUALITY GATE (esp. Clause 9, Type-3
SM-import π-audit) is binding; if any step fails the gate this probe
STOPS at `blocked` and NO predictions/ output files are produced.

CHECKPOINT-1 PRIOR-WORK (grep-before-scoping, mandatory):
  • Δα_had (hadronic vacuum polarization, ≈0.0277): `B1_QCD_HVP_
    substrate_scoping_2026-05-15.py` → **scoping NEGATIVE**, no clean
    K-rational analog, blocked on multiway formalism + R-14.  WALL.
  • parameter_linter.md Clause 9 canonical list EXPLICITLY names
    "Δα_had ≈ 0.0277" as a continuum-loop transcendental (∉ K by
    Lindemann); citing the imported Δα *value* as closure is K-INVALID.
  • So full-Δα is NOT bounded.  Only the leptonic part Δα_lep
    (R-14-free, pure QED) is even a candidate (9a) substrate analog.

PHYSICAL MAP (unified-oblique framework):
  photon = charge-weighted NEUTRAL/PERRON singlet (couples to Q, species-
  conserving, off the McKay support) → the resummed/Family-C side, the
  SAME channel as δ_r and S (NOT the h_P band-edge side).
  Δα = Π_γγ(M_Z²) − Π_γγ(0) = the Γ→P running ⇒ the S-type tree-cover
  neutral Γ→P flow [g(2√q) − g(k)], but charge-weighted (photon vertex
  ∝ Q) instead of S's c_S = 1/(2|E|) Perron-residue projection.

REFERENCE (Clause-2a; not a fit target — the K-analog must arise
first-principles, the number is only the post-hoc magnitude check):
  Δα_lep(M_Z²) ≈ 0.0314979   (3-loop QED, Steinhauser; very precise)
  in α^-1 units ≈ 0.0314979 × 137.036 ≈ 4.316

HARD PRE-DECLARED ABORTS (parameter_linter hard gate):
 (DA.1) Clause 9 — the SM Δα value is set by continuum loop integration
        with logs of lepton-mass ratios ln(M_Z²/m_ℓ²); these are
        transcendental over K.  The substrate analog MUST be a K-rational
        object DERIVED first-principles, NOT the imported number and NOT
        a K-fraction chosen because it lands near 0.0315.
 (DA.2) the photon-channel coefficient c_γ must be DERIVED from the
        framework's charge assignments + resolvent normalization (as
        c_S=1/(2|E|) was the Perron-residue projection), with NO freedom
        tuned to the Δα target.  Enumerate ONLY first-principles
        candidates; do not cherry-pick the closest.
 (DA.3) sign must be correct (Δα > 0: α grows from q²=0 to M_Z²).
 (DA.4) PASS only if a SINGLE first-principles K-rational c_γ × the
        tree-cover flow reproduces Δα_lep within the δ_r/δρ/S structural
        tolerance (~5% rel) WITHOUT fitting.  Otherwise → BLOCKED
        (Clause-9 resolution 9b: STRUCTURAL-DERIVATION-CONDITIONAL,
        named open mechanism, NOT theorem-grade; do NOT produce files).
"""
from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path
import sympy as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "predictions"))
from proofs.common import K_STAR, GIRTH  # noqa: E402
from alpha_1 import predict_alpha_1  # noqa: E402

k = K_STAR; q = k - 1; g = GIRTH
a1 = Fraction(q, k) ** (g - 2)                       # (2/3)^8
two_E = 12

# tree cavity g(z) (reuse the rigorous S-derivation object, unchanged)
z = sp.symbols('z', positive=True)
f_expr = (z - sp.sqrt(z**2 - 4*q)) / (2*q)
g_expr = 1 / (z - k*f_expr)
g_triv = sp.nsimplify(g_expr.subs(z, k))             # = 2/3  (Γ, off-support)
g_edge = sp.nsimplify(g_expr.subs(z, 2*sp.sqrt(q)))  # = √2   (on-cut repr.)
flow = sp.nsimplify(g_edge - g_triv)                 # √2 − 2/3
flow_f = float(flow)

DALPHA_LEP = 0.0314979                                # reference (NOT a fit target)

print("=" * 80)
print("  ATTEMPT — substrate Δα_lep via the photon channel of the one B_NB")
print("=" * 80)
print(f"  tree-cover neutral Γ→P flow  g(2√q)−g(k) = {flow} ≈ {flow_f:.6f}")
print(f"  Δα_lep reference (Clause-2a, post-hoc magnitude check only) = {DALPHA_LEP}")
print()

# --- (DA.2) FIRST-PRINCIPLES photon-coefficient candidates ONLY ------------
# The photon couples to Q.  Leptonic content the photon sees: 3 charged
# leptons, Q²=1 each, colour-singlet ⇒ ΣQ²_lep = 3.  Resolvent vertex
# normalisations available first-principles (same family as c_S, c=1/2):
#   c_S      = 1/(2|E|) = 1/12        (Perron-residue singlet projection)
#   1/(N·k*) = 1/12                   (handshake-equivalent)
#   1/k*, 1/(k*-1), 1/2 (W-norm), 1   (the established structural coeffs)
# Combined with ΣQ²_lep=3 and optionally the leading α₁ (n_fixed=2) or
# the resummed α₁/(1−α₁) (Perron channel, as δ_r/S use):
cand = {
    "ΣQ²·c_S·α₁/(1−α₁)":            3*Fraction(1,12)*a1/(1-a1),
    "ΣQ²·c_S·α₁":                   3*Fraction(1,12)*a1,
    "ΣQ²·(1/k*)·α₁/(1−α₁)":         3*Fraction(1,3)*a1/(1-a1),
    "c_S·α₁/(1−α₁)  [=S coeff]":    Fraction(1,12)*a1/(1-a1),
    "ΣQ²/(2|E|)  [no α₁]":          Fraction(3,12),
    "(1/k*)  [no α₁]":              Fraction(1,3),
    "α₁/(1−α₁)  [no c]":            a1/(1-a1),
    "ΣQ²·α₁  [no c]":               3*a1,
}
print("  first-principles c_γ candidates × flow  vs  Δα_lep (NO cherry-pick):")
print(f"  {'form':<34}{'c_γ':>14}{'c_γ·flow':>14}{'vs Δα_lep':>12}")
print("  " + "-"*74)
best = None
for label, cg in cand.items():
    val = float(cg) * flow_f
    off = (val - DALPHA_LEP)/DALPHA_LEP*100
    print(f"  {label:<34}{str(cg):>14}{val:>14.6f}{off:>+11.1f}%")
    if best is None or abs(off) < abs(best[2]):
        best = (label, val, off, cg)

print()
blabel, bval, boff, bcg = best
print(f"  closest first-principles form: {blabel}  → {bval:.6f}  ({boff:+.1f}% vs Δα_lep)")
print()

# --- verdict against pre-declared aborts -----------------------------------
print("=" * 80)
print("  VERDICT (parameter_linter hard quality gate)")
print("=" * 80)
within_struct = abs(boff) < 5.0          # δ_r/δρ/S structural tolerance
# (DA.1) Clause 9: is the *only* way to hit Δα_lep a fitted/cherry-picked
# K-fraction rather than a forced first-principles one?
forced_unique = False   # is there ONE structurally-forced c_γ (not "closest of 8")?
print(f"  (DA.1) Clause 9 — SM Δα_lep = Σ (α/3π)·ln(M_Z²/m_ℓ²): lepton-mass-")
print(f"         LOG structure, transcendental over K (Lindemann).  The")
print(f"         imported delta_alpha_running=9.092 is a Type-3 continuum-QED")
print(f"         number; citing it is K-INVALID.  A K-analog must be FORCED")
print(f"         first-principles, not selected for proximity.            → BINDING")
print(f"  (DA.2) photon coefficient: NO single c_γ is structurally FORCED —")
print(f"         the photon-vertex normalisation in the B_NB resolvent is")
print(f"         not derived (unlike c_S=1/(2|E|) Perron projection or")
print(f"         c=1/2 W-norm, which WERE forced).  The {len(cand)} candidates")
print(f"         span {min(float(c) for c in cand.values())*flow_f:.4f}–"
      f"{max(float(c) for c in cand.values())*flow_f:.4f}; 'closest' = cherry-pick.  → FAIL")
print(f"  (DA.3) sign: flow = √2−2/3 > 0 ⇒ candidates > 0 ⇒ Δα > 0 OK")
print(f"  (DA.4) PASS needs a FORCED K-rational c_γ within ~5%: ", end="")
print("NOT MET" if not forced_unique else "met")
print()
print("  → BLOCKED at the hard quality gate (Clause 9 + DA.2).")
print("    Reason: (i) Δα_had analog is B1-scoping-NEGATIVE (multiway+R-14")
print("    wall); (ii) Δα_lep has NO first-principles-FORCED K-rational")
print("    photon coefficient — the photon-vertex resolvent normalisation")
print("    is not derived the way c_S / c=1/2 were, so any K-fraction near")
print("    0.0315 would be cherry-picked (numerology gate failure); (iii)")
print("    the SM value is lepton-mass-log transcendental (Clause 9).")
print("    Linter protocol: STOP — do NOT produce predictions/Delta_alpha")
print("    files.  Resolution = Clause 9 (9b): tag delta_alpha_running as")
print("    STRUCTURAL-DERIVATION-CONDITIONAL / named open mechanism, NOT")
print("    theorem-grade.  This is not a failure — it protects downstream.")
print()
print("  HONEST PROPAGATION (the lesson, per user 'propagate everywhere'):")
print("   • delta_alpha_running=9.092 is the framework's ONLY un-derived")
print("     EM-running input — a Type-3 continuum-QED import (Clause 9).")
print("   • Per the clean-ratio diagnostic it is the SECONDARY piece of the")
print("     R_∞ residual (+0.007 in α^-1); the DOMINANT piece is the")
print("     α_EM(M_Z) gauge-cluster drift (−0.021 in α^-1).  Deriving Δα")
print("     would NOT close the clean ratio — the gauge-cluster drift is")
print("     the real lever.  This must be said wherever Δα touches.")
print("   • Touch-set to tag Clause-9-conditional: predictions/R_infinity.py")
print("     (the import), alpha_EM (the running framing), and the EM-cluster")
print("     rows that thread α-running (ledger / scorecard / master-doc).")
print("=" * 80)
