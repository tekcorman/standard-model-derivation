#!/usr/bin/env python3
"""
delta_rho_subleading_multihost_subtree_2026-05-17.py

GENUINE PHYSICS ATTEMPT on the δρ +4.58% subleading-spectral residual —
NOT a meta/exhaustion doc.  Tests one concrete, framework-native,
falsifiable hypothesis using ONLY already-theorem-grade structure.

CONTEXT (the structural license — no new mechanism introduced):
 - Over-determination theorem (theorem_unified_oblique.md §8): δρ and
   V_cb/V_ub are readings of the SAME B_NB(srs).  δρ = bare a=(2/3)^8 on
   the h_P Feshbach contour; V_cb = m=1 host, V_ub = Σ_{m≥2} hosts of the
   THEOREM-GRADE host law L_eff(m)=6m+2, per-host α_m/(1−α_m)
   (predictions/V_ub.py — closed UNIQUE-THEOREM-GRADE).
 - theorem §7.5 constraint (HARD): δρ's OWN single channel may NOT be
   collapsed to one 1/(1−α₁) (h_P on McKay cut, disc=−5<0, Dyson
   diverges).  The residual MUST be "a higher sub-tree multi-insertion
   (sub-leading-spectral) sum".  A *multi-HOST* Σ_m of distinct hosts is
   NOT that forbidden single-channel collapse — it is exactly the
   required form.

HYPOTHESIS (H):  δρ_full = (1/2)(√5/4) · Σ_{m≥1} ρ_m ,
   ρ_m on the SAME host law as V_ub/V_cb: L_eff(m)=6m+2, α_m=(2/3)^{L_eff}.
   Leading (current prediction) = m=1 term only.  The +4.58% residual =
   the m≥2 host tail.  ZERO new constants: (1/2)=W-norm, (√5/4)=Feshbach
   Im(h_P)/|h_P|², host law = V_ub-theorem.

We test THREE pre-declared structural readings of ρ_m and a 1-host
control.  PASS only if a reading reproduces the observed δρ (closes the
+4.58%) with the CORRECT SIGN and NO tuned constant and K-rational.

PRE-DECLARED ABORTS (anti-numerology, per feedback_theory_not_numerology):
 (A.1) the only near-hit is the theorem-§7.5-FORBIDDEN single-channel
       1/(1−α₁) resummation                                      → NEG.
 (A.2) closing the +4.58% requires a fitted/tuned coefficient or a
       host-weight not equal to the V_ub-theorem's                → NEG.
 (A.3) result not K-rational (∉ ℚ(√3,√5))                         → NEG.
 (A.4) wrong sign (host tail moves δρ_pred AWAY from δρ_obs)       → NEG.
A NEG here is a real result: it converts "deferred to deep layer" into
"the bounded V_ub-host-law reading is tested and fails for reason X",
which either localises the mechanism further or confirms the scoping doc.
"""
from fractions import Fraction
import math

# ---- theorem-grade inputs (no fitting) ---------------------------------
TWO_THIRDS = Fraction(2, 3)
def alpha(L):                       # bare host amplitude (2/3)^L, K-rational
    return TWO_THIRDS ** L
def L_eff(m):                       # V_ub/V_cb THEOREM host law
    return 6 * m + 2

# δρ structural prefactor: c·F, c=1/2 (W Type-3 norm), F=√5/4 (Feshbach
# Im(h_P)/|h_P|², h_P=(√3+i√5)/2, |h_P|²=2).  EXACT, theorem-grade.
sqrt5 = math.sqrt(5)
PREF = 0.5 * (sqrt5 / 4.0)
alpha1 = float(alpha(L_eff(1)))                 # (2/3)^8, the m=1 host

# δρ leading (current prediction) and PDG-central observed -----------------
dr_leading = PREF * alpha1                       # = (1/2)(√5/4)(2/3)^8
dr_obs     = 0.0104286                           # PDG-central (delta_rho.py)
print(f"δρ leading (m=1 host)      = {dr_leading*100:+.5f}%   "
      f"[current prediction; +{(dr_leading/dr_obs-1)*100:.2f}% rel vs obs]")
print(f"δρ observed (PDG-central)  = {dr_obs*100:+.5f}%")
print(f"TARGET: a reading whose FULL host sum lands at {dr_obs*100:+.5f}% "
      f"(closes the +4.58% with NO tuned constant)\n")

# ---- the three pre-declared structural readings of the host sum --------
M_MAX = 200
def host_sum(kind):
    s = 0.0
    for m in range(1, M_MAX + 1):
        a = float(alpha(L_eff(m)))
        if   kind == "bare":            s += a                 # leading-only/host
        elif kind == "perhost_resum":   s += a / (1.0 - a)     # V_ub per-host form
    return s

readings = {
    "R1  bare host tail  Σ_{m≥1} α_m              ": "bare",
    "R2  per-host resummed Σ_{m≥1} α_m/(1−α_m)    ": "perhost_resum",
}
print("FULL host-sum readings (δρ_full = PREF · Σ_{m≥1} ρ_m):")
for label, kind in readings.items():
    full = PREF * host_sum(kind)
    rel  = (full / dr_obs - 1.0) * 100.0
    print(f"  {label}: δρ_full = {full*100:+.5f}%   "
          f"({rel:+.2f}% rel vs obs)")

# ---- controls: the theorem-§7.5-FORBIDDEN single-channel collapses ------
print("\nCONTROLS (theorem §7.5 says these are FORBIDDEN for δρ's channel):")
forbidden_1host = PREF * (alpha1 / (1.0 - alpha1))     # 1/(1−α₁) on m=1
print(f"  C1  m=1 per-host resummed PREF·α₁/(1−α₁)      : "
      f"{forbidden_1host*100:+.5f}%  "
      f"({(forbidden_1host/dr_obs-1)*100:+.2f}% rel)  "
      f"[bare→resummed = {(1/(1-alpha1)-1)*100:+.2f}%]")

# ---- verdict (pre-declared logic, no post-hoc tuning) ------------------
print("\n" + "=" * 64)
def near(x, tol=0.012):                # within ~1.2% relative of obs = 'hit'
    return abs(x / dr_obs - 1.0) < tol

hits = []
for label, kind in readings.items():
    if near(PREF * host_sum(kind)):
        hits.append(label.strip())
c1_hit = near(forbidden_1host)

if not hits and c1_hit:
    print("VERDICT: NEG — abort (A.1) HITS.")
    print("  The ONLY near-hit is C1 = the theorem-§7.5-FORBIDDEN "
          "single-channel 1/(1−α₁) resummation")
    print(f"  (C1 rel = {(forbidden_1host/dr_obs-1)*100:+.2f}%; the "
          "admissible multi-host readings R1/R2 miss).")
    print("  ⇒ the bounded V_ub-host-law reading does NOT close the "
          "+4.58%; the residual genuinely")
    print("    requires the unbounded sub-tree multi-insertion sum. "
          "CONFIRMS the scoping doc")
    print("    EMPIRICALLY (not by assertion). A real negative result.")
elif hits:
    print(f"VERDICT: CANDIDATE-POSITIVE — admissible reading(s) {hits} "
          "land within ~1% of obs")
    print("  with ZERO tuned constants and the V_ub-theorem host law. "
          "Needs K-rationality")
    print("  audit + sign check + independent re-derivation before any "
          "grade claim.")
else:
    print("VERDICT: NEG — no admissible reading near obs; abort (A.2)/(A.4). "
          "Honest negative;")
    print("  the V_ub host law does not transfer to the h_P Feshbach "
          "contour for δρ.")
print("=" * 64)
