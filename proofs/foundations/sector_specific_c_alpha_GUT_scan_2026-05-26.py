#!/usr/bin/env python3
"""
Sector-specific α_GUT dark-correction probe: extract c_1, c_2, c_3 that close
the gauge cluster at PDG-σ, and compare against structurally plausible
rationals.

CONTEXT
-------
The framework currently applies a uniform dark correction at M_unif:

    α_GUT^obs = α_GUT^bare × (1 - c · α_1^bare/(1 - α_1^bare))    with c = 1/k* = 1/3

derived structurally in `theorem_alpha_GUT_dark_correction.md` (Routes H + C)
as the bipartite-marginal mode fraction (k*-2)/k* on srs (k*=3 gives 1/3).

The two-stage M_SUSY scan (`two_stage_RG_M_SUSY_scan_2026-05-26.py`) ruled out
TeV-scale threshold corrections as the source of the cluster residuals. The
next candidate is the *uniform-c assumption itself*: when the three SM gauge
sectors split below M_unif, each may carry a different fraction of the
bipartite marginal sector (different Wilson-loop H¹ content per sub-bundle).

This probe:
  1. Extracts c_i sector-by-sector by solving for what c value closes
     1/α_i(M_Z) at PDG.
  2. Compares against the uniform c = 1/3 baseline.
  3. Compares each c_i against plausible mode-count rationals n/(2|E|) = n/12
     and ratios involving k*.
  4. Reports a structural verdict.

If c_i ∈ [0, 1] and three c_i lie near distinct clean fractions n/12, the
residual pattern is structurally consistent with a sector-resolved Route H
mode count. If c_i go out of [0, 1] or land at irrational-looking numbers,
the uniform-c assumption is *not* the dominant residual source.

NOT theorem-grade. Diagnostic only. A positive result here is a *target*
for a structural derivation (re-running the Route H bipartite-marginal mode
count separately for U(1)_Y, SU(2)_L, SU(3)_c sub-bundles on srs), not a
derivation itself.
"""

import math
import os
import sys
from fractions import Fraction

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'predictions'))

from predictions.M_unif import predict_M_unif_GeV
from predictions.M_Z import M_Z_GeV as _M_Z
from predictions.M_Pl_natural import M_Pl_GeV

# ------------------------------------------------------------------
# Framework inputs (all theorem-grade or theorem-grade-conditional)
# ------------------------------------------------------------------
K_STAR        = 3
G_GIRTH       = 10
ALPHA_GUT_BARE = 1.0/(2**K_STAR * K_STAR)     # = 1/24
ALPHA_1_BARE  = (2.0/3.0)**(G_GIRTH - 2)        # = (2/3)^8 = 256/6561
X_DARK        = ALPHA_1_BARE / (1.0 - ALPHA_1_BARE)   # = 256/6305
M_UNIF        = float(predict_M_unif_GeV(K_STAR, G_GIRTH, M_Pl_GeV))
M_Z           = float(_M_Z)
HYP_NORM      = 3.0 / 5.0

# β-coefficients (single-regime MSSM as currently used; framework's choice)
B_MSSM = (33.0/5.0, 1.0, -3.0)

LOG_RATIO = math.log(M_Z / M_UNIF)              # negative

# ------------------------------------------------------------------
# PDG cluster (mirror framework's σ_PDG reporting)
# ------------------------------------------------------------------
PDG_sin2W      = (0.23121, 0.00004)
PDG_alpha_EM   = (1.0/127.944, 0.014/127.944**2)
PDG_alpha_s    = (0.11800, 0.00090)

# Self-consistent PDG triplet (1/α_1, 1/α_2, 1/α_3) derived from (sin²W, α_EM, α_s)
# to avoid the small g_i↔(sin²W, α_EM) inconsistency in PDG-listed bundles.
def pdg_inv_alphas():
    s2 = PDG_sin2W[0]
    aEM = PDG_alpha_EM[0]
    aS  = PDG_alpha_s[0]
    a2 = aEM / s2
    aY = aEM / (1.0 - s2)
    a1 = aY / HYP_NORM
    return (1.0/a1, 1.0/a2, 1.0/aS)

# Approximate σ on each inverse coupling (error propagation, leading-order)
def pdg_sigma_inv():
    s2, ss2 = PDG_sin2W
    aEM, saEM = PDG_alpha_EM
    aS, saS = PDG_alpha_s
    a2 = aEM / s2
    # σ(1/a2) ~ σ(a2)/a2² with σ(a2) ≈ √((saEM/s2)² + (aEM·ss2/s2²)²)
    sa2 = math.hypot(saEM/s2, aEM*ss2/s2/s2)
    sigma_inv_a2 = sa2 / a2**2
    aY = aEM / (1.0 - s2)
    saY = math.hypot(saEM/(1.0 - s2), aEM*ss2/(1.0 - s2)**2)
    a1 = aY / HYP_NORM
    sa1 = saY / HYP_NORM
    sigma_inv_a1 = sa1 / a1**2
    sigma_inv_a3 = saS / aS**2
    return (sigma_inv_a1, sigma_inv_a2, sigma_inv_a3)


# ------------------------------------------------------------------
# Core extraction: given target 1/α_i(M_Z) = T_i, find c_i
# ------------------------------------------------------------------
def c_for_sector(target_inv_alpha_at_MZ, b_i):
    """Solve for the c value such that the dark-corrected α_GUT_i, when run
    down with b_i, lands at the target 1/α at M_Z."""
    inv_alpha_GUT_obs = target_inv_alpha_at_MZ + (b_i / (2.0*math.pi)) * LOG_RATIO
    # 1/α_GUT_obs = (1/α_GUT_bare) / (1 - c x), invert:
    ratio = ALPHA_GUT_BARE * inv_alpha_GUT_obs    # = (1 - c x)^{-1}
    one_minus_cx = 1.0 / ratio
    c = (1.0 - one_minus_cx) / X_DARK
    return c, inv_alpha_GUT_obs


def predict_inv_alpha_at_MZ(c_i, b_i):
    """Forward direction: given c_i, predict 1/α_i(M_Z).
    α_GUT^obs = α_GUT^bare × (1 - c·x), so 1/α_GUT^obs = 1/(α_GUT^bare × (1-c·x))."""
    inv_alpha_GUT_obs = 1.0 / (ALPHA_GUT_BARE * (1.0 - c_i * X_DARK))
    return inv_alpha_GUT_obs - (b_i / (2.0*math.pi)) * LOG_RATIO


# ------------------------------------------------------------------
# Closest n/12 and other plausible rationals
# ------------------------------------------------------------------
def closest_twelfth(c):
    """Closest n/12 (n integer in [0, 12])."""
    best_n = round(c * 12)
    best_n = max(0, min(12, best_n))
    return Fraction(best_n, 12), c - best_n/12.0

PLAUSIBLE_RATIONALS = [
    # (label, value)
    ('(k*-2)/k* = 1/3', Fraction(K_STAR - 2, K_STAR)),                 # current uniform
    ('(k*-2)/(k*+1) = 1/4', Fraction(K_STAR - 2, K_STAR + 1)),
    ('(k*-1)/(2k*) = 1/3', Fraction(K_STAR - 1, 2*K_STAR)),
    ('1/2', Fraction(1, 2)),
    ('5/12', Fraction(5, 12)),                                          # v_Higgs c
    ('1/4', Fraction(1, 4)),
    ('3/8', Fraction(3, 8)),
    ('1/6', Fraction(1, 6)),
    ('(k*-2)/(2k*) = 1/6', Fraction(K_STAR - 2, 2*K_STAR)),
]

def best_rational(c, tol=0.015):
    """Closest member of PLAUSIBLE_RATIONALS within tol; else None."""
    best = None
    best_d = float('inf')
    for label, fr in PLAUSIBLE_RATIONALS:
        d = abs(c - float(fr))
        if d < best_d:
            best_d = d
            best = (label, fr, d)
    if best_d <= tol:
        return best
    return None


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
print("="*78)
print("  Sector-specific α_GUT dark correction c_i probe  (2026-05-26)")
print("="*78)
print(f"  α_GUT^bare = 1/{2**K_STAR * K_STAR} = {ALPHA_GUT_BARE:.6f}")
print(f"  α_1^bare   = (2/3)^{G_GIRTH-2} = {ALPHA_1_BARE:.6f}")
print(f"  x = α_1/(1-α_1) = 256/6305 = {X_DARK:.6f}")
print(f"  Current uniform c = (k*-2)/k* = 1/{K_STAR} = {1.0/K_STAR:.6f}")
print(f"  M_unif = {M_UNIF:.4e} GeV,  M_Z = {M_Z:.4f} GeV")
print(f"  ln(M_Z/M_unif) = {LOG_RATIO:.4f}")
print()

# PDG targets (1/α at M_Z) — self-consistent triplet
pdg_inv = pdg_inv_alphas()
pdg_sig = pdg_sigma_inv()
print("PDG self-consistent targets at M_Z (from sin²W, α_EM, α_s):")
for i, (v, s) in enumerate(zip(pdg_inv, pdg_sig), start=1):
    print(f"  1/α_{i}(M_Z) = {v:8.4f}  ± {s:.4f}")
print()

# Baseline (uniform c = 1/3)
print("-"*78)
print("BASELINE — uniform c = 1/3 (current framework):")
print("-"*78)
c_uniform = 1.0/K_STAR
for i, (b, inv_target, sig) in enumerate(zip(B_MSSM, pdg_inv, pdg_sig), start=1):
    pred = predict_inv_alpha_at_MZ(c_uniform, b)
    nsig = (pred - inv_target) / sig
    print(f"  sector {i}:  predicted 1/α = {pred:8.4f},   target = {inv_target:8.4f},   "
          f"Δ = {pred-inv_target:+.4f}  ({nsig:+.2f}σ)")
print()

# Sector-specific c extraction
print("-"*78)
print("SECTOR-SPECIFIC c EXTRACTION — what c_i closes each sector at PDG?")
print("-"*78)
extracted = []
for i, (b, inv_target) in enumerate(zip(B_MSSM, pdg_inv), start=1):
    c_i, inv_alpha_GUT_obs = c_for_sector(inv_target, b)
    extracted.append((i, c_i, inv_alpha_GUT_obs))
    cf, dt = closest_twelfth(c_i)
    rat = best_rational(c_i)
    rat_str = f"  ≈ {rat[0]} (Δ={rat[2]:+.4f})" if rat else "  (no clean rational within 0.015)"
    print(f"  sector {i}:  c_{i} = {c_i:.5f}    1/α_GUT^obs_{i} = {inv_alpha_GUT_obs:.4f}    "
          f"closest n/12 = {cf} (Δ={dt:+.4f}){rat_str}")
print()

c1, c2, c3 = (e[1] for e in extracted)

# In-range and ordering check
in_range = all(0.0 <= c <= 1.0 for c in (c1, c2, c3))
print("-"*78)
print("STRUCTURAL PLAUSIBILITY CHECK")
print("-"*78)
print(f"  All c_i in [0, 1]?                          {'YES' if in_range else 'NO'}")
print(f"  Monotone in i (c_1 > c_2 > c_3 or reverse)? "
      f"{'YES (c_1>c_2>c_3)' if c1 > c2 > c3 else 'YES (c_3>c_2>c_1)' if c3>c2>c1 else 'NO (non-monotone)'}")
print(f"  Spread c_1 - c_3 = {c1-c3:+.4f}")
print(f"  Distance of uniform-c (1/3) from each: |c_1 - 1/3| = {abs(c1-1/3):.4f}, "
      f"|c_2 - 1/3| = {abs(c2-1/3):.4f}, |c_3 - 1/3| = {abs(c3-1/3):.4f}")
print()

# Forward check: plug in c values, recompute all 6 observables
print("-"*78)
print("FORWARD CHECK — predict cluster with extracted c_i")
print("-"*78)
inv_a1 = predict_inv_alpha_at_MZ(c1, B_MSSM[0])
inv_a2 = predict_inv_alpha_at_MZ(c2, B_MSSM[1])
inv_a3 = predict_inv_alpha_at_MZ(c3, B_MSSM[2])
a1, a2, a3 = 1.0/inv_a1, 1.0/inv_a2, 1.0/inv_a3
aY = HYP_NORM * a1
s2  = aY/(a2 + aY)
aEM = a2 * s2
g_i = [math.sqrt(4*math.pi*a) for a in (a1, a2, a3)]

PDG_cluster = {
    'sin²θ_W(M_Z)':  (PDG_sin2W[0],   PDG_sin2W[1]),
    'g_1(M_Z)':      (0.46144,        0.00010),
    'g_2(M_Z)':      (0.65200,        0.00010),
    'g_3(M_Z)':      (1.21800,        0.00500),
    'α_s(M_Z)':      (PDG_alpha_s[0], PDG_alpha_s[1]),
    '1/α_EM(M_Z)':   (127.944,        0.014),
}
preds = {
    'sin²θ_W(M_Z)':  s2,
    'g_1(M_Z)':      g_i[0],
    'g_2(M_Z)':      g_i[1],
    'g_3(M_Z)':      g_i[2],
    'α_s(M_Z)':      a3,
    '1/α_EM(M_Z)':   1.0/aEM,
}
print(f"  {'observable':>14}  {'predicted':>10}  {'PDG':>10}  {'Nσ_PDG':>8}")
for k in ['sin²θ_W(M_Z)', 'g_1(M_Z)', 'g_2(M_Z)', 'g_3(M_Z)', 'α_s(M_Z)', '1/α_EM(M_Z)']:
    pv = preds[k]
    pdg_v, pdg_s = PDG_cluster[k]
    nsig = (pv - pdg_v) / pdg_s
    print(f"  {k:>14}  {pv:>10.5f}  {pdg_v:>10.5f}  {nsig:+8.2f}")
print()

# Final diagnosis
print("="*78)
print("DIAGNOSIS")
print("="*78)
if not in_range:
    print(f"  c_i fall outside [0, 1]: c=({c1:.3f}, {c2:.3f}, {c3:.3f})")
    print("  → Sector-specific c cannot be derived from a positive mode-count fraction.")
    print("  → Sector-specific dark correction RULED OUT as residual source.")
else:
    # Are any c_i within 0.015 of a clean rational?
    rationals = [best_rational(c) for c in (c1, c2, c3)]
    clean_count = sum(1 for r in rationals if r is not None)
    print(f"  c_i extraction: c_1 = {c1:.4f},  c_2 = {c2:.4f},  c_3 = {c3:.4f}")
    print(f"  Clean-rational matches (within 0.015):  {clean_count} of 3")
    for i, r in enumerate(rationals, start=1):
        if r is not None:
            print(f"    c_{i} ≈ {r[0]}  (Δ = {r[2]:+.4f})")
        else:
            print(f"    c_{i}: no plausible rational nearby")
    print()
    if clean_count >= 2:
        print("  → c_i lie near clean fractions of |2E|=12 (the Hashimoto NB total dim).")
        print("    Structurally plausible: a Route-H mode count resolved by sub-bundle")
        print("    (U(1)_Y / SU(2)_L / SU(3)_c) instead of by unified gauge sector.")
        print("    → Target: redo bipartite-marginal mode count *per gauge sub-bundle*")
        print("      on srs and check whether the predicted c_i match these values.")
    else:
        print("  → c_i don't land at clean structural rationals.")
        print("    → Residual source is structurally elsewhere; sector-specific c is")
        print("      a possible *partial* fix but not the natural mechanism.")
print()
print("Probe outputs are diagnostic only. Sector-specific c values become a")
print("THEOREM-GRADE move only after substrate-derived (not fit-derived) c_i are")
print("calculated from the Hashimoto sub-bundle dimension count.")
print("="*78)
