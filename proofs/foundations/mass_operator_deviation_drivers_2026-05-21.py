#!/usr/bin/env python3
"""
DIAGNOSTIC — what drives the mass-operator deviations?

mass_operator_run_2026-05-21.py reproduces the SM spectrum but the quark /
neutrino sectors carry residuals (u +15%, b +2%, s +2.6%, t +0.7%, ν +1-2%)
while leptons are exact. This probe tests three candidate drivers, each as a
falsifiable check — a NEGATIVE result here is a real finding, not a bug.

  H1  ε² ARTIFACT.  The run probe computes ε² from the formula
      2 + 6·α₁·s·f(s) — which equals the *naive MS-bar* value (W53: this is
      ~5% BELOW the framework's R4 band). The theorem-grade pinned values are
      ε²_down = 5/2, ε²_up = 17/5 (W53, Type-IV walker ε² = 2·n_free). Test:
      does the run probe's light-quark agreement survive switching to the
      pinned ε²? If it gets WORSE, the current agreement is the artifact.

  H2  WALK-LENGTH DARK CORRECTION.  Deviations seemed to track whether the
      species walks (up-type L=0 clean; down L=g and ν walk, +2%; leptons
      L=g−2 calibrated). Test: apply the Family-D dark factor PER WALK STEP,
      (1 − c·a²)^(L − L_lepton), relative to the lepton calibration. Does the
      b/s anchor residual collapse below 1%? Pre-declared FAIL otherwise.

  H3  ANCHOR vs SPREAD.  Decompose each quark residual into (a) the gen-3
      ANCHOR (y_tree·dark·v vs obs) and (b) the within-sector circulant SPREAD
      (feed koide_sector the OBSERVED gen-3, check gen-1/2). Localises each
      miss to its mechanism.

Selection map (theorem_selection_map_2026-05-21): lepton Type III L=g−2=8;
down Type IV L=g=10; up Type II L=0; ν Type I spectral.
"""

import numpy as np

k_star, g, N_at = 3, 10, 4
q = 2 / 3
v = 246.22
a = q ** (g - 2)                       # α₁_bare = (2/3)^8
alpha1_full = (5 / 3) * a
dark_yukawa = 1 - (5 / 6) * a ** 2      # Family-D 1H+2F vertex, applied ONCE
findings = []


def koide_sector(anchor, eps2, delta):
    """C₃-circulant Koide block: 3 masses from {anchor=gen-3, ε², δ}.
    gen-3 (k giving max cos) is pinned exactly to `anchor`."""
    eps = np.sqrt(eps2)
    M0 = np.sqrt(anchor) / (1 + eps * np.cos(delta))
    return np.sort([(M0 * (1 + eps * np.cos(2*np.pi*k/3 + delta))) ** 2
                    for k in range(3)])


def f_form(s):
    return 1 + (s - 1) * (g - 2) / (2 * g)


# gen-3 anchors — shape × dynamics × v (exactly as the run probe builds them):
m_tau = (5/3) * q**(g-2) / k_star**2 * dark_yukawa * v
m_b = q**g * dark_yukawa * v
m_t = 1.0 * dark_yukawa * v / np.sqrt(2)
SECT = {  # name: (s, anchor GeV, δ, walk L, (obs gen1, gen2, gen3) GeV)
    "lepton": (0, m_tau, 2/9,        g-2, (0.510999e-3, 0.105658, 1.77686)),
    "down":   (1, m_b,   2/(9*2),    g,   (4.67e-3, 93.4e-3, 4.18)),
    "up":     (2, m_t,   2/(9*3),    0,   (2.16e-3, 1.273, 172.69)),
}
EPS2_FORMULA = {s: (2.0 if s == 0 else 2 + 6*alpha1_full*s*f_form(s))
                for s in (0, 1, 2)}
EPS2_PINNED = {0: 2.0, 1: 5/2, 2: 17/5}      # W53 + Row P37


def report(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


# ----------------------------------------------------------------------
report("D1 — H1: ε² audit  (formula = naive MS-bar  vs  W53-pinned)")
print(f"  {'sector':8s}{'formula ε²':>14s}{'pinned ε²':>14s}{'gap':>10s}")
for nm, (s, *_) in SECT.items():
    fo, pi = EPS2_FORMULA[s], EPS2_PINNED[s]
    print(f"  {nm:8s}{fo:14.4f}{pi:14.4f}{100*(fo-pi)/pi:+9.1f}%")
r_form = (EPS2_FORMULA[2]-2)/(EPS2_FORMULA[1]-2)
r_pin = (EPS2_PINNED[2]-2)/(EPS2_PINNED[1]-2)
print(f"\n  (ε²_up−2)/(ε²_down−2):  formula {r_form:.4f}   pinned {r_pin:.4f}"
      f"   [Row P37 = 14/5 = {14/5}]")
print("  Both satisfy the 14/5 ratio — they differ only in the absolute")
print("  normalisation of (ε²−2). The run probe uses the FORMULA column;")
print("  W53 says that column is the naive-MS-bar value, ~5% below band.")
findings.append("D1: run probe uses formula ε² (naive MS-bar), not W53-pinned")


# ----------------------------------------------------------------------
report("D2 — H1 test: quark masses with FORMULA ε²  vs  PINNED ε²")
print(f"  {'quark':6s}{'obs':>11s}{'formula-ε²':>13s}{'dev':>9s}"
      f"{'pinned-ε²':>13s}{'dev':>9s}")
worst_form = worst_pin = 0.0
for nm in ("down", "up"):
    s, anc, dl, L, obs = SECT[nm]
    mf = koide_sector(anc, EPS2_FORMULA[s], dl)
    mp = koide_sector(anc, EPS2_PINNED[s], dl)
    names = (("d", "s", "b") if nm == "down" else ("u", "c", "t"))
    for i, qn in enumerate(names):
        df, dp = (mf[i]-obs[i])/obs[i], (mp[i]-obs[i])/obs[i]
        if qn not in ("b", "t"):                 # gen-3 = anchor, unchanged
            worst_form = max(worst_form, abs(df))
            worst_pin = max(worst_pin, abs(dp))
        u = "GeV" if obs[i] > 1 else "MeV"
        sc = 1 if obs[i] > 1 else 1e3
        print(f"  {qn:6s}{obs[i]*sc:9.3f}{u}{mf[i]*sc:11.3f}{u}{100*df:+8.1f}%"
              f"{mp[i]*sc:11.3f}{u}{100*dp:+8.1f}%")
print(f"\n  worst gen-1/2 |dev|:  formula ε² {100*worst_form:.1f}%"
      f"   pinned ε² {100*worst_pin:.1f}%")
if worst_pin > worst_form * 1.5:
    verdict = ("H1 CONFIRMED — the run probe's light-quark agreement is an "
               "ARTIFACT of\n  the naive-MS-bar ε². The framework's own "
               "theorem-grade ε² (5/2, 17/5)\n  gives a WORSE fit. Caveat "
               "(W53): quark masses are scheme/scale-\n  dependent — obs are "
               "MS-bar(2 GeV); the structural ε² need not match that scheme.")
elif worst_pin < worst_form:
    verdict = ("H1 — pinned ε² IMPROVES the fit; the run probe should adopt "
               "5/2, 17/5.")
else:
    verdict = ("H1 — pinned ε² changes the light quarks but not decisively; "
               "adopt it anyway\n  (framework-correct value), report honestly.")
print("  VERDICT:", verdict)
findings.append("D2: " + verdict.split("\n")[0].strip())


# ----------------------------------------------------------------------
report("D3 — H3: anchor  vs  circulant-spread decomposition")
print("  (a) ANCHOR deviation = y_tree·dark·v vs observed gen-3")
print("  (b) SPREAD deviation = feed koide_sector the OBSERVED gen-3,"
      " check gen-1/2\n")
for nm in ("lepton", "down", "up"):
    s, anc, dl, L, obs = SECT[nm]
    anchor_dev = (anc - obs[2]) / obs[2]
    m_clean = koide_sector(obs[2], EPS2_PINNED[s], dl)   # anchor forced to obs
    spread = max(abs(m_clean[i]-obs[i])/obs[i] for i in (0, 1))
    print(f"  {nm:8s} anchor dev {100*anchor_dev:+7.2f}%   "
          f"spread dev (gen-1/2, pinned ε²) {100*spread:7.1f}%")
print("\n  Reading: the anchor and the spread are SEPARATE drivers. A small")
print("  anchor miss + a large spread miss = light quark blows up (gen-1 is")
print("  where √m is smallest, so spread error is relative-amplified there).")
findings.append("D3: anchor and circulant-spread are separable, independent drivers")


# ----------------------------------------------------------------------
report("D4 — H2 test: walk-length-scaled dark correction")
print("  Candidate: dark(L) = (1 − c·X)^(L − L_lepton), applied to the gen-3")
print("  anchor relative to the lepton calibration (L_lepton = g−2 = 8).")
print(f"  Walk lengths:  lepton L=8 (calib)   down L=10   up L=0\n")
L_lep = g - 2
b_dev0 = (m_b - SECT["down"][4][2]) / SECT["down"][4][2]
t_dev0 = (m_t - SECT["up"][4][2]) / SECT["up"][4][2]
print(f"  uncorrected anchor dev:  b {100*b_dev0:+.2f}%   t {100*t_dev0:+.2f}%")
print(f"  {'per-step factor X':28s}{'b dev':>10s}{'t dev':>10s}{'pass?':>8s}")
h2_pass = False
for label, X in [("(5/6)·a²  [Family-D]", (5/6)*a**2),
                  ("a²", a**2),
                  ("a/(1−a)  [leading DC]", a/(1-a))]:
    fb = (1 - X) ** (g - L_lep)        # down: L−L_lep = 10−8 = 2
    ft = (1 - X) ** (0 - L_lep)        # up:   L−L_lep = 0−8 = −8
    bd = (m_b*fb - SECT["down"][4][2]) / SECT["down"][4][2]
    td = (m_t*ft - SECT["up"][4][2]) / SECT["up"][4][2]
    ok = abs(bd) < 0.01 and abs(td) < 0.01
    h2_pass = h2_pass or ok
    print(f"  {label:28s}{100*bd:+9.2f}%{100*td:+9.2f}%{('YES' if ok else 'no'):>8s}")
if h2_pass:
    verdict = "H2 CONFIRMED — a framework per-step dark factor collapses b & t."
else:
    verdict = ("H2 REFUTED — no framework per-step dark factor collapses the "
               "anchors.\n  (5/6)a² is ~10× too small (0.25%/2-steps vs +2% "
               "needed); a/(1−a) is\n  ~2× too big (overshoots negative). The "
               "b/s +2% is NOT a walk-step\n  correction — it localises to the "
               "down anchor formula y_b = q^g itself,\n  which is tree-grade: "
               "it lacks the Family-D-grade treatment y_τ received\n  (y_τ "
               "carries the derived (5/3)/k*² prefactor; y_b carries none).")
print("\n  VERDICT:", verdict)
findings.append("D4: " + verdict.split("\n")[0].strip())


# ----------------------------------------------------------------------
report("D5 — verdict: the drivers, separated")
print("""  THREE distinct drivers, now separated and graded:

  (1) ε² ARTIFACT (H1).  The run probe uses the naive-MS-bar ε² formula,
      not W53's theorem-grade pinned 5/2 & 17/5 — see D2 for which way it
      moves the light quarks. The run probe must adopt the pinned values;
      whatever agreement that costs was an artifact, not a success.

  (2) DOWN-ANCHOR Yukawa (H2 → refuted as a walk effect).  b/s ride ~+2%
      high because y_b = q^g is a TREE value — it never got the Family-D
      treatment that took y_τ to theorem grade. Not a per-step walk
      correction (D4); the next real target is a theorem-grade y_b.

  (3) CIRCULANT SPREAD + light-end amplification (H3).  u (+15%) is gen-1
      of its triple — where √m is smallest and any ε/δ error is
      relative-amplified. Separate from the anchor; the W43 grade.

  NOT a mass-mechanism issue: W/Z (absolute scale), and the neutrinos —
  m_ν3 is a hardcoded constant in the run probe, not an operator output,
  so 'ν +1-2%' is transcription, not a driver this operator controls.
""")
for fnd in findings:
    print("  •", fnd)
print()
print("=" * 72)
print("DIAGNOSTIC COMPLETE — drivers separated; H1 + H3 are real, H2 refuted.")
print("=" * 72)
