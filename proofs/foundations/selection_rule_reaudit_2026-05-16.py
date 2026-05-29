#!/usr/bin/env python3
"""
proofs/foundations/selection_rule_reaudit_2026-05-16.py

SELECTION-RULE RE-AUDIT.

The §6.1 work (`theorem_unified_oblique.md` §7.5) DERIVED the
propagator-level dark-correction form-selection rule from the analytic
structure of the tree cavity resolvent g(z)=1/(z−k·f(z)).  Previously
the master-doc family taxonomy (`theorem_substrate_feshbach_dark_
corrections_master.md` §5) was assigned by observable-class HEURISTICS
+ the v_Higgs c=5/12 calibration anchor.  Now that the rule is DERIVED,
re-audit every propagator-level catalogue member for form↔channel
consistency.  A genuine misassignment would CHANGE that parameter's
predicted number (real propagation); confirmation is a rigor upgrade
for the whole tree-level-coupling sector.

THE DERIVED CRITERION, stated rigorously via the Ihara adjacency map
λ = h + q/h  (q ≡ k*−1 = 2; NB eigenvalue h ↔ adjacency eigenvalue λ):

  • OFF the McKay support  |λ| > 2√q  ⇔  disc ≡ λ²−4q > 0 :
      the tree resolvent is analytic there ⇒ the geometric (Dyson)
      resummation CONVERGES ⇒ **resummed Family-C  c·α₁/(1−α₁)**.
  • ON the McKay cut       |λ| ≤ 2√q  ⇔  disc = λ²−4q ≤ 0
      (Ramanujan-saturated modes |h|=√q map ONTO the cut):
      √disc imaginary ⇒ no convergent geometric resummation ⇒
      **leading-only Family-E / Feshbach  ∝ α₁**.

SHARPENING (surfaced by this audit): the committed §7.5 used the
band edge z=2√q (disc=0) as the representative on-cut point.  The
GENERAL criterion is disc ≤ 0 (the whole cut).  The δρ channel h_P
maps to INTERIOR λ=√3 (disc=−5<0) — on the cut but NOT at the edge.
Still ⇒ leading.  The §7.5 wording must say "on the McKay cut
(disc ≤ 0)", not "the band edge", else it mis-states δρ's location.

SCOPE: this criterion governs PROPAGATOR-level corrections (the
resummed-vs-leading form choice) ONLY.  Family-D (vertex per-leg
multiway dark-disruption, ∝ α₁²) is a DIFFERENT mechanism (master
doc §3 D) and is OUT OF SCOPE — the criterion must NOT be applied to
y_τ / λ_Higgs.  "No dark correction" members (V_us, V_cb, …) likewise.

PRE-DECLARED ABORTS:
 (A.1) the λ=h+q/h map mis-classifies a CANONICAL anchor — δ_r must
       come out off-support/resummed, δρ on-cut/leading             → audit invalid.
 (A.2) a MISASSIGNMENT is found (a propagator member whose assigned
       form contradicts its derived λ-location)                      → REAL
       finding: that parameter's number would change; report it, do
       NOT paper over.
 (A.3) all propagator members consistent                              → taxonomy
       CONFIRMED (rigor upgrade, ZERO numerical churn) + report the
       sharpening + the δρ-resummation-forbidden corollary.  Do NOT
       manufacture a reassignment to claim numerical impact.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import K_STAR, h_P  # noqa: E402

k = K_STAR
q = k - 1                       # 2
support_edge = 2 * sp.sqrt(q)   # 2√2  (McKay support |λ| ≤ 2√q)

print("=" * 84)
print("  SELECTION-RULE RE-AUDIT — derived §6.1 criterion vs master-doc taxonomy")
print("=" * 84)
print()
print(f"  k*={k}, q=k*-1={q}, McKay support |λ| ≤ 2√q = {float(support_edge):.5f}")
print(f"  criterion:  disc = λ²−4q > 0  ⇒ OFF-support ⇒ RESUMMED Family-C α₁/(1−α₁)")
print(f"              disc = λ²−4q ≤ 0  ⇒ ON the cut  ⇒ LEADING  Family-E/Feshbach ∝α₁")
print()


def lam(h):
    """Ihara adjacency eigenvalue λ = h + q/h for NB eigenvalue h."""
    return sp.simplify(h + q / h)


def classify(h):
    L = lam(h)
    disc = sp.simplify(L * sp.conjugate(L) - 4 * q) if sp.im(L) != 0 else sp.simplify(L**2 - 4*q)
    # On-cut iff |λ| ≤ 2√q  (equivalently disc ≤ 0 for real λ; for λ on the
    # cut Im part ≠ 0 ⇒ Ramanujan-saturated ⇒ on-cut by construction).
    absL2 = sp.simplify(sp.Abs(L)**2)
    on_cut = bool(absL2 <= 4 * q)
    return L, sp.simplify(absL2 - 4*q), on_cut


# ---------------------------------------------------------------------------
# Part 0 — verify the map on the canonical anchors  (A.1)
# ---------------------------------------------------------------------------
print("=" * 84)
print("Part 0 — canonical anchors (pre-declared abort A.1)")
print("=" * 84)
print()
h_triv = sp.Integer(1)          # trivial/symmetric rep  (A·1 = k·1 ⇔ h=1)
h_marg = sp.Integer(-1)         # marginal |λ|=1 NB sector (h=±1)
h_ram = sp.Rational(1, 1) * sp.sqrt(q)   # real Ramanujan band-edge rep h=√q
hP = (sp.sqrt(3) + sp.I*sp.sqrt(5)) / 2  # the framework eigenvalue h_P

for name, h in [("trivial h=1", h_triv), ("marginal h=-1", h_marg),
                ("band-edge h=√q", h_ram), ("h_P (=(√3+i√5)/2)", hP)]:
    L, dd, oc = classify(h)
    loc = "ON cut (leading)" if oc else "OFF support (resummed)"
    print(f"  {name:<22}: λ = {sp.nsimplify(L)!s:<14}  |λ|²−4q = {dd}  ⇒ {loc}")

L_dr, _, oc_dr = classify(h_triv)            # δ_r channel ≈ trivial/Perron
L_drho, _, oc_drho = classify(hP)            # δρ channel = h_P
A1 = (not oc_dr) and oc_drho                  # δ_r off-support, δρ on-cut
print()
print(f"  (A.1) δ_r channel (trivial/Perron) OFF-support: {not oc_dr};  "
      f"δρ channel (h_P) ON-cut: {oc_drho}  ⇒ map valid: {A1}")
assert A1, "(A.1) canonical anchors misclassified — audit invalid"
print(f"  NOTE: h_P → λ = {sp.nsimplify(L_drho)} (= √3, INTERIOR of the cut, "
      f"|λ|²−4q = {sp.simplify(sp.Abs(L_drho)**2-4*q)} < 0) — NOT the band")
print(f"  edge (disc=0).  The committed §7.5 'band edge z=2√q' wording is the")
print(f"  disc=0 representative; the general criterion is disc ≤ 0 (whole cut).")

# ---------------------------------------------------------------------------
# Part 1 — audit the master-doc §5 propagator-level catalogue
# ---------------------------------------------------------------------------
print()
print("=" * 84)
print("Part 1 — master-doc §5 catalogue audit")
print("=" * 84)
print()

# (member, assigned form, substrate channel → representative h, scope)
CAT = [
    ("v_Higgs",      "Family-C resummed c·α₁/(1−α₁) (c=5/12)",
     "marginal/Route-H (h=±1)",        h_marg, "propagator"),
    ("α_GUT",        "Family-C resummed c·α₁/(1−α₁) (c=1/k*)",
     "cycle-marginal Stark-Terras (h=±1)", h_marg, "propagator"),
    ("δ_r  (M_Z)",   "Family-C resummed c_S·α₁/(1−α₁) (c_S=1/12)",
     "Perron/neutral (h=1)",           h_triv, "propagator"),
    ("S   (oblique)","Family-C resummed (tree-cover g-flow)",
     "neutral Perron tree (h=1)",      h_triv, "propagator"),
    ("δρ  (ρ-param)","Family-E leading c·F·α₁ (F=√5/4)",
     "h_P phase",                      hP,     "propagator"),
    ("m_ν3",         "Feshbach leading (mechanism baked in, no extra 1/(1−α₁))",
     "spectral-gap h (Ramanujan)",     hP,     "propagator"),
    ("β cosmic bir.","Berry leading sin(arg h), c=1 (no resummation)",
     "arg(h_P) phase",                 hP,     "propagator"),
    ("θ_23 PMNS",    "leading tan²(arg h)=5/3",
     "arg(h) phase",                   hP,     "propagator"),
    ("U   (oblique)","leading ≈0 (Ramanujan sector scale-frozen)",
     "√q Ramanujan sector",            h_ram,  "propagator"),
    ("y_τ",          "Family-D per-leg −(5/6)α₁²",
     "VERTEX (1H+2F)",                 None,   "vertex"),
    ("λ_Higgs",      "Family-D per-leg −4α₁²",
     "VERTEX (4H)",                    None,   "vertex"),
    ("V_us",         "direct A2 counting 9/40 (NO dark correction)",
     "—",                              None,   "no-DC"),
    ("Λ_CC w_eff",   "V_Ram h↔h̄ split (distinct mechanism)",
     "—",                              None,   "distinct"),
]

def predicted_form(h):
    _, _, oc = classify(h)
    return "leading (Family-E/Feshbach)" if oc else "resummed (Family-C)"

def assigned_is_resummed(form_str):
    return "resummed" in form_str.lower() or "α₁/(1−α₁)" in form_str

print(f"  {'member':<14}{'scope':<11}{'channel→λ loc':<22}{'predicted':<26}{'verdict'}")
print("  " + "-" * 95)
misassigned = []
for member, assigned, chan, h, scope in CAT:
    if scope != "propagator":
        print(f"  {member:<14}{scope:<11}{'—':<22}{'(criterion N/A)':<26}OUT OF SCOPE ✓")
        continue
    L, dd, oc = classify(h)
    pred = predicted_form(h)
    pred_resummed = (not oc)
    assigned_resummed = assigned_is_resummed(assigned)
    consistent = (pred_resummed == assigned_resummed)
    loc = f"λ={sp.nsimplify(L)} ({'on-cut' if oc else 'off-supp'})"
    tag = "CONSISTENT ✓" if consistent else "*** MISASSIGNED ***"
    if not consistent:
        misassigned.append((member, assigned, pred))
    print(f"  {member:<14}{scope:<11}{loc:<22}{pred:<26}{tag}")

# ---------------------------------------------------------------------------
# Part 2 — verdict + propagating corollaries
# ---------------------------------------------------------------------------
print()
print("=" * 84)
print("Part 2 — verdict (pre-declared aborts)")
print("=" * 84)
print()
if misassigned:
    print(f"  (A.2) MISASSIGNMENT(S) FOUND — REAL numerical-impact finding:")
    for m, a, p in misassigned:
        print(f"     {m}: assigned [{a}] but derived criterion ⇒ {p}")
    print(f"  These parameters' predicted numbers would CHANGE on correction.")
else:
    print(f"  (A.3) NO misassignment.  Every propagator-level catalogue member's")
    print(f"  assigned form is CONSISTENT with the independently-derived spectral")
    print(f"  criterion.  The taxonomy — previously observable-class heuristics +")
    print(f"  the v_Higgs c=5/12 calibration anchor — is now DERIVED-CONSISTENT.")
    print(f"  Numerical impact: ZERO (no reassignment; not manufactured).")
    print(f"  Rigor impact: the resummed-vs-leading form choice for the WHOLE")
    print(f"  tree-level-coupling sector is now grounded in the cavity resolvent's")
    print(f"  analytic structure, not heuristics.")
print()
print(f"  Non-trivial confirmation: v_Higgs (the c=5/12 calibration ANCHOR)")
print(f"  comes out OFF-support/resummed PURELY from λ(h=±1)=±k off the McKay")
print(f"  support — the derived criterion agrees with the empirical anchor it")
print(f"  was never told about.  α_GUT likewise (cycle-marginal h=±1 → λ=±k).")
print()
print(f"  SHARPENING (correctness fix to committed §7.5): the rule is")
print(f"  disc ≤ 0 (on the McKay cut), NOT 'the band edge z=2√q'.  h_P sits")
print(f"  at INTERIOR λ=√3 (|λ|²−4q = -5 < 0), on the cut but not at the")
print(f"  edge; it still ⇒ leading.  §7.5 wording to be sharpened.")
print()
print(f"  δρ COROLLARY (propagates to the open +4.58% problem): δρ's channel")
print(f"  h_P is ON the cut (λ=√3, disc<0) ⇒ the geometric resummation does")
print(f"  NOT converge there ⇒ adding a 1/(1−α₁) to δρ to absorb the +4.58%")
print(f"  is FORBIDDEN by the derived criterion.  The +4.58% MUST be a")
print(f"  higher sub-tree multi-insertion (sub-leading-spectral) sum, not a")
print(f"  missed resummation.  This constrains how that residual can close.")
print()
print(f"  SCOPE held: Family-D (y_τ, λ_Higgs) is vertex per-leg α₁² — a")
print(f"  DIFFERENT mechanism, correctly OUT OF SCOPE (criterion not applied).")
print()
print("=" * 84)
print("End of selection-rule re-audit.")
print("=" * 84)
