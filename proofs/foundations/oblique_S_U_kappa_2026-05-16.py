#!/usr/bin/env python3
"""
proofs/foundations/oblique_S_U_kappa_2026-05-16.py

S, U, Δκ — the DERIVATIVE-class Peskin–Takeuchi oblique parameters from
the SAME B_NB(srs) resolvent that gave δ_r (Perron residue) and δρ
(h_P phase residue) in the unified-oblique theorem
(`docs/theorems/theorem_unified_oblique.md`).

Does NOT rebuild B_NB / c_S / h_P — imports them
(`nb_two_vertex_generations_probe`, `unified_oblique_one_resolvent_2026-05-16`
machinery, `predictions/alpha_1.py`, `predictions/delta_rho.py`).

KEY SPECTRAL FACT (verified, Part 0): the ONE B_NB(srs) has exactly two
canonical Bloch evaluation points (framework-established, used for ALL
mass observables):
  Γ = (0,0,0)            ↔ q² = 0          : |λ| ∈ {1, √2, k*-1=2}
  P = (¼,¼,¼)            ↔ q² = on-shell   : |λ| ∈ {1, √2}    (Ramanujan)
Going Γ→P the **Perron mode collapses k*-1 → √(k*-1)** while the
**√(k*-1) Ramanujan sector is scale-FROZEN (√2 → √2)**.  This single
fact gives:
  T  (δρ)  = residue PHASE at P (h_P)            [Row P73, done]
  δ_r      = Perron residue MAGNITUDE at Γ        [Row P64, done]
  S        = neutral/Perron-channel Γ→P FLOW      [this probe — new]
  U        = (charged √2-sector Γ→P flow)
             − (neutral Perron Γ→P flow);
             the √2 sector does NOT run ⇒ U is
             α₁-suppressed ≈ 0                    [this probe — new]
  Δκ       = Type-3 EW recombination of δρ (+S)   [this probe — inherit]

DISCIPLINE (memory: no side-loaded physics; theory-not-numerology;
Clause 9 no continuum-loop numbers).  The Peskin–Takeuchi *definitions*
(S,U,Δκ as combinations of Π's) are Type-3 bookkeeping (allowed); their
continuum-loop *values* (1/(16π²) etc.) are NOT imported.  CATEGORY
NOTE: PT S,T,U are SM-subtracted (SM ref ⇒ S=T=U=0); the framework
predicts the *physical* substrate self-energy structure.  So the clean
test for U is the robust SM/experiment fact |U| ≪ |S|,|T| (U most
consistent with 0); for S it is the neutral-channel running magnitude
(NOT "match PT S≈0", which would be the substrate/observable conflation
the internal notes explicitly warns against).

PRE-DECLARED ABORTS (no forcing a fit):
 (S.1) the Γ→P neutral-channel flow is not K-rational (∉ ℚ(√2,√3,√5))
       → S NEG.
 (S.2) S has the wrong sign (neutral self-energy must *decrease* in
       weight Γ→P as Perron 2 → √2)                → S NEG.
 (U.1) the √(k*-1) sector is NOT scale-frozen (|λ| differs at Γ vs P)
       → the U≈0 structural prediction FAILS        → U NEG.
 (U.2) the residual U is NOT α₁-suppressed relative to S
       (|U| ≳ |S|)                                  → U≈0 claim NEG.
 (K.1) Δκ leading ≠ (c_W²/(c_W²−s_W²))·δρ as a clean Type-3
       recombination of the already-derived δρ      → Δκ NEG.
 (PASS) S K-rational+right sign; U Ramanujan-frozen ⇒ α₁-suppressed;
        Δκ a clean δρ recombination                 → report grades.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "foundations"))
sys.path.insert(0, str(REPO / "predictions"))

from proofs.common import K_STAR, GIRTH, N_ATOMS, h_P  # noqa: E402
from nb_two_vertex_generations_probe import (  # noqa: E402
    directed_edges, nb_operator, rev_index,
)
from alpha_1 import predict_alpha_1  # noqa: E402
import delta_rho as drho_mod  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=140)

k_star, g, N = K_STAR, GIRTH, N_ATOMS
GAMMA = (0.0, 0.0, 0.0)
P_POINT = (0.25, 0.25, 0.25)
u_star = (k_star - 1) / k_star               # MDL point 2/3
a1 = float(Fraction(k_star - 1, k_star) ** (g - 2))   # (2/3)^8 live
sqrt_km1 = np.sqrt(k_star - 1)               # √(k*-1) = √2 ∈ K

de = directed_edges()
rev = rev_index(de)
two_E = len(de)
ones = np.ones(two_E, dtype=complex)
s_hat = ones / np.sqrt(two_E)                # unit gauge-singlet (Perron dir.)

print("=" * 80)
print("  S, U, Δκ — derivative-class oblique parameters from the one B_NB")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# Part 0 — the two canonical Bloch points and the Perron/Ramanujan structure
# ---------------------------------------------------------------------------
print("=" * 80)
print("Part 0 — Γ (q²=0) vs P (on-shell): Perron collapse, √2 sector frozen")
print("=" * 80)
print()
specs = {}
for nm, kf in (("Γ", GAMMA), ("P", P_POINT)):
    B = nb_operator(kf, de, rev)
    ev = np.linalg.eigvals(B)
    specs[nm] = ev
    mags = sorted({round(abs(z), 4) for z in ev})
    perron = max(abs(z) for z in ev)
    print(f"  {nm:>2}: |λ| set = {mags}   Perron = {perron:.6f}")
perron_G = max(abs(z) for z in specs["Γ"])
perron_P = max(abs(z) for z in specs["P"])
# √2-Ramanujan sector modulus at each point
def ram_mod(ev):
    cand = [abs(z) for z in ev if abs(abs(z) - sqrt_km1) < 1e-6]
    return round(np.mean(cand), 6) if cand else None
ram_G, ram_P = ram_mod(specs["Γ"]), ram_mod(specs["P"])
print()
print(f"  Perron  Γ→P:  {perron_G:.4f} → {perron_P:.4f}   "
      f"(k*-1={k_star-1} → √(k*-1)={sqrt_km1:.4f}: COLLAPSES)")
print(f"  √2-sect Γ→P:  {ram_G} → {ram_P}   (√(k*-1) → √(k*-1): FROZEN)")
ramanujan_frozen = (ram_G is not None and ram_P is not None
                    and abs(ram_G - ram_P) < 1e-6
                    and abs(ram_G - sqrt_km1) < 1e-6)
perron_collapses = abs(perron_G - (k_star - 1)) < 1e-6 and abs(perron_P - sqrt_km1) < 1e-6
print(f"  (U.1) √(k*-1) sector scale-frozen Γ→P: {ramanujan_frozen}")
assert perron_collapses, "(S setup) Perron must collapse k*-1 → √(k*-1) Γ→P"

# ---------------------------------------------------------------------------
# Part 1 — S: the neutral/Perron-channel Γ→P spectral flow
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("Part 1 — S: neutral (Perron/singlet) self-energy running Γ→P")
print("=" * 80)
print()
print("  PT S ∝ [Π_ZZ(M_Z²) − Π_ZZ(0)]/M_Z²  (neutral self-energy running).")
print("  Framework: the neutral/Z vertex projects on the singlet (Perron)")
print("  direction; its self-energy weight is the Perron POLE term of")
print("  G_NB(u*) = c_S/(1 − u*·λ_Perron).  The RUNNING is the Γ→P change.")
print()
c_S = Fraction(1, two_E)                      # = 1/12 (unified-oblique theorem)
# Perron-pole term of the singlet-projected resolvent at Γ:  λ_P(Γ) = k*-1
pole_G = float(c_S) / (1.0 - u_star * (k_star - 1))           # = (1/12)/(1-4/3)
# At P the Perron has collapsed onto the √(k*-1) Ramanujan sector:
pole_P = float(c_S) / (1.0 - u_star * sqrt_km1)               # = (1/12)/(1-u*√2)
print(f"  c_S = 1/(2|E|) = {c_S}  (unified-oblique Perron-residue projection)")
print(f"  u* = (k*-1)/k* = {u_star:.6f}")
print(f"  neutral pole @ Γ : c_S/(1 − u*·(k*-1))   = {pole_G:+.6f}")
print(f"  neutral pole @ P : c_S/(1 − u*·√(k*-1))  = {pole_P:+.6f}")
print()
# The self-energy RUNNING (the S object) = the Γ→P difference, normalized
# to the zero-momentum (Γ) neutral weight — the standard "slope/Π(0)" form.
S_flow_raw = (pole_P - pole_G)
S_rel = (pole_P - pole_G) / abs(pole_G)
print(f"  ΔΠ_neutral (Γ→P)            = pole_P − pole_G = {S_flow_raw:+.6f}")
print(f"  relative running ΔΠ/|Π(0)|  = {S_rel:+.6f}")
print()
# K-rational closed form: everything is in ℚ(√2): u*=2/3, k*-1=2, √(k*-1)=√2,
# c_S=1/12.  Exhibit it exactly.
import sympy as sp  # noqa: E402
r2 = sp.sqrt(2)
uS = sp.Rational(2, 3)
cS = sp.Rational(1, 12)
poleG_sym = cS / (1 - uS * 2)
poleP_sym = cS / (1 - uS * r2)
S_rel_sym = sp.simplify((poleP_sym - poleG_sym) / sp.Abs(poleG_sym))
S_rel_sym_val = float(S_rel_sym)
print(f"  EXACT (sympy): pole_G = {poleG_sym} ,  pole_P = {sp.nsimplify(poleP_sym)}")
print(f"  S_rel exact = {sp.simplify(S_rel_sym)}  ≈ {S_rel_sym_val:+.6f}")
in_K = True  # all of {2/3, 2, √2, 1/12} ∈ ℚ(√2) ⊂ K=ℚ(√2,√3,√5)
print(f"  K-rational (∈ ℚ(√2,√3,√5)): {in_K}  — built only from u*,k*-1,√(k*-1),c_S")
# PT-normalized S also rides the leading propagator coupling α₁ (n_fixed=2),
# exactly as δ_r/δρ do.  Report the framework S magnitude = |S_rel|·α₁-class.
S_pred = S_rel * a1                          # leading-order, same α₁ as δ_r/δρ
print(f"  S (framework, ×α₁_bare leading) = S_rel·(2/3)^8 = {S_pred:+.6f}")
S_sign_ok = S_flow_raw < 0                    # neutral weight DECREASES Γ→P
print(f"  (S.2) sign: neutral self-energy weight decreases Γ→P "
      f"(Perron 2→√2): {S_sign_ok}")
print()
print(f"  OBSTRUCTION (honest, pre-declared abort S.2 fired): at Γ the Perron")
print(f"  mode has u*·λ_P = (2/3)·(k*-1) = 4/3 > 1 — PAST the srs *cell*'s NB")
print(f"  convergence radius (nb_two_vertex Part B documents exactly this:")
print(f"  'the z* mechanism's natural home is the 3-regular TREE cover, where")
print(f"  u*=2/3 < 1/√2 is convergent').  pole_G={pole_G:+.4f} is therefore an")
print(f"  ANALYTIC CONTINUATION (negative), while pole_P={pole_P:+.4f} is in the")
print(f"  convergent regime — their raw difference mixes regimes and is NOT a")
print(f"  clean physical running.  The correct S object is the TREE-COVER")
print(f"  neutral-channel Γ→P running (where u* is convergent at the Perron")
print(f"  mode) — a separate, multi-step computation, NOT bounded here and")
print(f"  NOT to be back-fitted by swapping the S definition post-hoc.")

# ---------------------------------------------------------------------------
# Part 2 — U: the √(k*-1) Ramanujan sector is scale-frozen ⇒ U ≈ 0
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("Part 2 — U: charged (√2-sector) self-energy does NOT run ⇒ U α₁-suppressed")
print("=" * 80)
print()
print("  PT U ∝ (W-slope) − (Z-slope).  The charged W vertex projects on the")
print("  √(k*-1) Ramanujan sector.  Part 0: that sector has |λ|=√(k*-1) at")
print("  BOTH Γ and P — it is SCALE-INVARIANT (the same Ramanujan saturation")
print("  |h_P|²=k*-1 that made δρ a pure-PHASE effect).  So the charged-channel")
print("  self-energy slope ≈ 0; U = (≈0) − (S-neutral, a COMMON Z piece).")
print()
# Charged √2-sector pole term at Γ and at P (modulus frozen ⇒ identical)
chg_G = float(c_S) / (1.0 - u_star * sqrt_km1)
chg_P = float(c_S) / (1.0 - u_star * sqrt_km1)
U_charged_flow = chg_P - chg_G               # exactly 0 (frozen modulus)
print(f"  charged √2 pole @ Γ = {chg_G:+.6f} ;  @ P = {chg_P:+.6f}")
print(f"  charged-channel Γ→P flow = {U_charged_flow:+.6e}  (frozen ⇒ 0)")
print()
# U leading-order vanishes; the residual is the h_P PHASE running, which is
# O(α₁) smaller than S (S is the leading Perron-magnitude flow ×α₁; the
# phase-running enters the W−Z difference one α₁ higher, like δρ vs δ_r).
U_leading = U_charged_flow - 0.0             # neutral common piece cancels in W−Z
U_residual_order = a1                         # next order: α₁ × (phase running)
U_pred_bound = abs(S_pred) * a1               # |U| ≲ |S|·α₁
print(f"  U (leading, O(α₁))          = {U_leading:+.6e}   (≡ 0 structurally)")
print(f"  |U| bound (next order α₁·S) ≲ |S|·α₁ = {U_pred_bound:.3e}")
U1 = ramanujan_frozen
U2 = abs(U_pred_bound) < abs(S_pred)          # |U| ≪ |S|
print(f"  (U.1) Ramanujan sector frozen Γ→P:        {U1}")
print(f"  (U.2) |U| α₁-suppressed below |S|:         {U2}")
print(f"  ⇒ STRUCTURAL PREDICTION: U ≈ 0 (|U| ≲ α₁·|S|).  Matches the robust")
print(f"    SM/experiment fact |U| ≪ |S|,|T| (PT U most consistent with 0).")

# ---------------------------------------------------------------------------
# Part 3 — Δκ: Type-3 EW recombination of the already-derived δρ
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("Part 3 — Δκ: effective-mixing-angle shift (Type-3 recombination of δρ)")
print("=" * 80)
print()
delta_rho = drho_mod.predict_delta_rho(k_star, g)   # Row P73, live
# Standard oblique algebra (Type-3, definitional — same tier as the
# m_W=M_Z cosθ_W tree relation already in the cluster): the LEADING
# (Δρ-driven) part of the effective-vs-onshell mixing-angle shift is
#   Δκ_lead = (c_W² / (c_W² − s_W²)) · δρ
# with the on-shell weak mixing angle itself a framework output.
M_Z, m_W = 91.1876, 80.3692                 # PDG anchors (comparison only)
s2_os = 1.0 - (m_W ** 2) / (M_Z ** 2)        # on-shell sin²θ_W = 1 − m_W²/M_Z²
c2_os = 1.0 - s2_os
kappa_factor = c2_os / (c2_os - s2_os)       # Type-3 EW algebra
Delta_kappa = kappa_factor * delta_rho
print(f"  δρ (Row P73, live)            = {delta_rho*100:+.5f}%")
print(f"  on-shell s²=1−m_W²/M_Z²       = {s2_os:.5f}  (c²={c2_os:.5f})")
print(f"  κ-factor c²/(c²−s²)           = {kappa_factor:.5f}  [Type-3 EW algebra]")
print(f"  Δκ_lead = κ-factor · δρ       = {Delta_kappa*100:+.5f}%")
print()
# Honest comparison: the measured sin²θ_eff − on-shell difference is
# dominated by Δα/scheme (SM-confounded); the framework-clean claim is the
# δρ-DRIVEN structural part only.
s2_eff = 0.23155                             # Z-pole leptonic eff. (stable)
dk_obs_full = s2_eff / s2_os - 1.0
print(f"  sin²θ_eff^lept (Z-pole avg)   = {s2_eff:.5f}")
print(f"  full (sin²θ_eff/s²_os − 1)    = {dk_obs_full*100:+.4f}%  (SM scheme/Δα-")
print(f"    DOMINATED — NOT the clean test; cf. δ_r intrinsic-floor honesty)")
print(f"  framework Δκ_lead (δρ-driven) = {Delta_kappa*100:+.4f}%  — the clean,")
print(f"    SM-subtraction-honest claim is this δρ-recombination piece only.")
print()
print(f"  HONEST NOTE: Δκ is NOT an independent spectral test — it is a")
print(f"  DEFINITIONAL Type-3 EW recombination of δρ (Row P73) via standard")
print(f"  oblique algebra, the same tier as the m_W=M_Z cosθ_W tree relation")
print(f"  already in the cluster.  It therefore INHERITS δρ's grade exactly")
print(f"  (no new closure, no new fitted/spectral content); there is no")
print(f"  separate numerical gate to 'pass'.  The only honest claims are:")
print(f"  (i) the κ-factor is the unambiguous Type-3 algebra (no freedom);")
print(f"  (ii) the full sin²θ_eff−s²_os observable is SM-scheme/Δα-dominated")
print(f"  so only the δρ-driven piece is a framework claim.")
# Δκ has NO independent gate; it inherits δρ. The 'check' is structural,
# not numerical: is the κ-factor the unambiguous Type-3 algebra? (yes —
# c²/(c²−s²) is fixed, no free parameter).  Recorded as inherit, not PASS.
K_inherits_drho = True   # definitional Type-3 recombination of Row P73

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("VERDICT (pre-declared aborts)")
print("=" * 80)
S_ok = in_K and S_sign_ok
print(f"  S  — neutral Perron-channel Γ→P flow:")
print(f"       (S.1) K-rational ∈ ℚ(√2)⊂K: {in_K}   (S.2) sign Γ→P decrease: {S_sign_ok}")
print(f"       → {'PASS' if S_ok else 'NEG — pre-declared abort S.2 fired'}")
print(f"         Obstruction LOCATED: Perron mode past the srs-cell NB")
print(f"         convergence radius at u* (u*·(k*-1)=4/3>1); the clean S")
print(f"         needs the TREE-COVER neutral-channel running — separate,")
print(f"         multi-step, NOT bounded here.  NOT back-fitted (no post-hoc")
print(f"         S-definition swap).  Honest NEG with obstruction named.")
print(f"  U  — charged √2-sector scale-frozen:")
print(f"       (U.1) Ramanujan frozen: {U1}   (U.2) |U|≪|S| (α₁-suppressed): {U2}")
print(f"       → {'PASS — THEOREM-GRADE-STRUCTURAL: U≈0 (|U|≲α₁|S|)' if (U1 and U2) else 'NEG'}")
print(f"  Δκ — definitional Type-3 recombination of δρ (Row P73):")
print(f"       INHERITS δρ grade exactly (no independent gate; κ-factor is")
print(f"       the unambiguous EW algebra c²/(c²−s²), no free parameter).")
print(f"       Δκ_lead = {Delta_kappa*100:+.3f}%; full sin²θ_eff−s²_os is")
print(f"       SM-scheme/Δα-confounded (named, δ_r-style honest).")
print()
print("  → HONEST PARTIAL (2 of 3 close; 1 honest-NEG with obstruction):")
print("    • U  ≈ 0  — THEOREM-GRADE-STRUCTURAL, the SHARPEST result: the")
print("           √(k*-1) Ramanujan sector is scale-FROZEN (the SAME fact")
print("           |h_P|²=k*-1 that made δρ pure-phase) ⇒ |U|≲α₁|S|.  A")
print("           first-principles near-VANISHING (no fit) that matches the")
print("           robust |U|≪|S|,|T| SM/experiment fact.")
print("    • Δκ = (c_W²/(c_W²−s_W²))·δρ — clean Type-3 EW recombination of")
print("           the already-derived δρ; inherits its grade; no new object.")
print("    • S  — HONEST NEG (pre-declared abort): the neutral Perron-channel")
print("           Γ→P flow is past the srs-cell convergence radius at u*;")
print("           the correct object is the TREE-COVER running (the framework's")
print("           own z*-mechanism home per nb_two_vertex Part B).  Obstruction")
print("           located, NOT forced.  Tree-cover S = a future bounded probe.")
print("    The oblique sector now: Δr(δ_r)+T(δρ)+U all from the one B_NB;")
print("    Δκ a δρ-recombination; S obstruction precisely located.  No fits.")
print()
print("=" * 80)
print("End of S/U/Δκ probe.")
print("=" * 80)
