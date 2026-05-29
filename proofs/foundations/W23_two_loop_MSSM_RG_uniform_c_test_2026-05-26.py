#!/usr/bin/env python3
"""
W23 — Path 4: Two-loop MSSM RG test with uniform c = 1/3

CONTEXT
-------
W21 + W22 (2026-05-26 EOD+1) showed that the natural substrate paths to
sector-specific c (c_color = 1/4, c_EW = 1/3) face structural obstructions:
  - W21: BS-T sector split within V_scalar is non-canonical at the graph level
  - W22: canonical SU(3)_c lift does not commute with Hashimoto B

Yesterday's `sector_specific_c_alpha_GUT_scan_2026-05-26.py` extracted from
PDG cluster:
  c_1 = 0.343 (Δ from 1/3 ≈ +0.010, 1.33σ on 1/α_1)
  c_2 = 0.332 (Δ from 1/3 ≈ -0.002, 0.27σ on 1/α_2)
  c_3 = 0.241 (Δ from 1/3 ≈ -0.092, 1.42σ on 1/α_3)

The EXTRACTION uses ONE-LOOP MSSM RG running. The c_3 = 0.241 reading is
heavily dominated by the +1.42σ residual on 1/α_3 at M_Z.

HYPOTHESIS (Path 4)
-------------------
The c_3 ≠ c_1 = c_2 pattern might be one-loop-running ARTIFACT, NOT
structural sector-specificity. Two-loop MSSM RG corrections to α_3
typically shift 1/α_3(M_Z) by O(0.1-1) units, comparable to the +0.092
one-loop residual.

If two-loop closes the α_3 residual with uniform c = 1/3, then:
  • sector-specific c is NOT needed
  • the framework's existing uniform-c theorem stands
  • W21/W22 obstructions are non-issues (no sector-specific c to derive)

If two-loop does NOT close α_3 with uniform c = 1/3, then:
  • sector-specific c is genuinely structural
  • we need a different mechanism (W21/W22 paths still open as research)

METHOD
------
Two-loop MSSM RGE (Martin SUSY primer 1997, Eq. 6.30; standard ref):
$$
\frac{d \alpha_i^{-1}}{dt} = -\frac{b_i^{(1)}}{2\pi} - \frac{1}{8\pi^2} \sum_j b_{ij}^{(2)} \alpha_j(t)
$$
with $t = \ln(\mu)$.

One-loop MSSM coefficients (already in framework):
$$b^{(1)} = (33/5, 1, -3)$$

Two-loop MSSM coefficients (from Martin Eq. 6.30):
$$b^{(2)} = \begin{pmatrix} 199/25 & 27/5 & 88/5 \\ 9/5 & 25 & 24 \\ 11/5 & 9 & 14 \end{pmatrix}$$

Integrate downward from M_unif to M_Z with boundary
$\alpha_i(M_unif) = \alpha_{GUT}^{obs}$ (uniform across sectors, dark-corrected
with c = 1/3).

Compare to PDG self-consistent triplet (1/α_1, 1/α_2, 1/α_3) at M_Z.

REFERENCES
----------
- Martin, S.P. (1997). A Supersymmetry Primer. hep-ph/9709356, §6.4 Eq. 6.30
  (two-loop gauge β-functions for MSSM with one Higgs doublet pair).
"""

import math
import os
import sys
from fractions import Fraction
import numpy as np
from scipy.integrate import solve_ivp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'predictions'))

from predictions.M_unif import predict_M_unif_GeV
from predictions.M_Z import M_Z_GeV as _M_Z
from predictions.M_Pl_natural import M_Pl_GeV

# ============================================================
# Framework inputs (theorem-grade or theorem-grade-conditional)
# ============================================================
K_STAR        = 3
G_GIRTH       = 10
ALPHA_GUT_BARE = 1.0 / (2**K_STAR * K_STAR)            # = 1/24
ALPHA_1_BARE  = (2.0/3.0)**(G_GIRTH - 2)                # = (2/3)^8 = 256/6561
X_DARK        = ALPHA_1_BARE / (1.0 - ALPHA_1_BARE)     # = 256/6305
C_UNIFORM     = 1.0 / K_STAR                            # = 1/3

# Dark-corrected α_GUT^obs (uniform c = 1/3, theorem-grade-cond per 2026-05-15)
ALPHA_GUT_OBS = ALPHA_GUT_BARE * (1.0 - C_UNIFORM * X_DARK)
INV_ALPHA_GUT_OBS = 1.0 / ALPHA_GUT_OBS

M_UNIF        = float(predict_M_unif_GeV(K_STAR, G_GIRTH, M_Pl_GeV))
M_Z           = float(_M_Z)

HYP_NORM      = 3.0 / 5.0  # 1/α_1 (GUT) = (3/5) · 1/α_Y (SM)

# MSSM β-coefficients
b1_MSSM = np.array([33.0/5.0, 1.0, -3.0])

# Two-loop MSSM matrix (Martin 1997 §6.4 Eq. 6.30)
b2_MSSM = np.array([
    [199.0/25.0, 27.0/5.0,  88.0/5.0],
    [9.0/5.0,    25.0,      24.0],
    [11.0/5.0,   9.0,       14.0]
])

# ============================================================
# PDG self-consistent targets at M_Z
# ============================================================
# Derive 1/α_1, 1/α_2, 1/α_3 from (sin²θ_W, α_EM, α_s) for self-consistency.
PDG_sin2W      = (0.23121, 0.00004)
PDG_alpha_EM   = (1.0/127.944, 0.014/127.944**2)
PDG_alpha_s    = (0.11800, 0.00090)

def pdg_inv_alphas():
    s2 = PDG_sin2W[0]
    aEM = PDG_alpha_EM[0]
    aS  = PDG_alpha_s[0]
    a2 = aEM / s2
    aY = aEM / (1.0 - s2)
    a1 = aY / HYP_NORM
    return np.array([1.0/a1, 1.0/a2, 1.0/aS])

def pdg_sigma_inv():
    s2, ss2 = PDG_sin2W
    aEM, saEM = PDG_alpha_EM
    aS, saS = PDG_alpha_s
    a2 = aEM / s2
    sa2 = math.hypot(saEM/s2, aEM*ss2/s2/s2)
    sigma_inv_a2 = sa2 / a2**2
    aY = aEM / (1.0 - s2)
    saY = math.hypot(saEM/(1.0 - s2), aEM*ss2/(1.0 - s2)**2)
    a1 = aY / HYP_NORM
    sa1 = saY / HYP_NORM
    sigma_inv_a1 = sa1 / a1**2
    sigma_inv_a3 = saS / aS**2
    return np.array([sigma_inv_a1, sigma_inv_a2, sigma_inv_a3])

# ============================================================
# RG running — both one-loop and two-loop
# ============================================================
def one_loop_rge(t, inv_a):
    """d(1/α_i)/dt = -b_i^(1) / (2π)  (one-loop, MSSM)."""
    return -b1_MSSM / (2.0 * math.pi)

def two_loop_rge(t, inv_a):
    """d(1/α_i)/dt = -b_i^(1)/(2π) - (1/(8π²)) Σ_j b_ij^(2) α_j  (two-loop, MSSM).

    Note: integrate WITH respect to t = ln(μ). We start at μ = M_unif (high)
    and run DOWN to μ = M_Z (low), so t decreases.
    """
    alpha = 1.0 / inv_a
    one_loop = -b1_MSSM / (2.0 * math.pi)
    two_loop = -(1.0 / (8.0 * math.pi**2)) * (b2_MSSM @ alpha)
    return one_loop + two_loop

def run_rge(rge_fn, alpha_GUT_unif, M_unif, M_Z):
    """Integrate the RGE from M_unif (high) DOWN to M_Z (low).

    Boundary: 1/α_i(M_unif) = 1/α_GUT_unif (uniform across sectors).

    Returns array [1/α_1(M_Z), 1/α_2(M_Z), 1/α_3(M_Z)].
    """
    t_unif = math.log(M_unif)
    t_Z = math.log(M_Z)
    # Initial: all three couplings equal at M_unif
    inv_a_init = np.full(3, 1.0 / alpha_GUT_unif)
    # Solve from t_unif down to t_Z (negative direction)
    sol = solve_ivp(rge_fn, (t_unif, t_Z), inv_a_init,
                    method='RK45', rtol=1e-10, atol=1e-12,
                    dense_output=False)
    if not sol.success:
        raise RuntimeError(f"RG integration failed: {sol.message}")
    return sol.y[:, -1]

# ============================================================
# Run both one-loop and two-loop predictions
# ============================================================
print("="*78)
print(" W23 — Two-loop MSSM RG with uniform c = 1/3 (Path 4)")
print("="*78)
print()
print(f" Inputs:")
print(f"   α_GUT^bare    = 1/{2**K_STAR * K_STAR} = {ALPHA_GUT_BARE:.6f}")
print(f"   α_1^bare      = (2/3)^{G_GIRTH-2} = {ALPHA_1_BARE:.6f}")
print(f"   x = α_1/(1-α_1) = {X_DARK:.6f}")
print(f"   Uniform c     = (k*-2)/k* = 1/{K_STAR} = {C_UNIFORM:.6f}")
print(f"   α_GUT^obs     = {ALPHA_GUT_OBS:.6f}  (1/α_GUT^obs = {INV_ALPHA_GUT_OBS:.4f})")
print(f"   M_unif        = {M_UNIF:.4e} GeV")
print(f"   M_Z           = {M_Z:.4f} GeV")
print(f"   ln(M_Z/M_unif) = {math.log(M_Z/M_UNIF):.4f}")
print()

pdg = pdg_inv_alphas()
pdg_sig = pdg_sigma_inv()
print(" PDG self-consistent targets at M_Z (from sin²W, α_EM, α_s):")
for i, (v, s) in enumerate(zip(pdg, pdg_sig), start=1):
    print(f"   1/α_{i}(M_Z) = {v:8.4f}  ± {s:.4f}")
print()

# One-loop prediction (current framework)
one_loop_pred = run_rge(one_loop_rge, ALPHA_GUT_OBS, M_UNIF, M_Z)
print("-"*78)
print(" ONE-LOOP MSSM RG (current framework):")
print("-"*78)
for i, (pred, tgt, sig) in enumerate(zip(one_loop_pred, pdg, pdg_sig), start=1):
    delta = pred - tgt
    nsig = delta / sig
    pct = delta / tgt * 100
    print(f"   1/α_{i}(M_Z):  predicted = {pred:8.4f},   PDG = {tgt:8.4f},   "
          f"Δ = {delta:+.4f}  ({nsig:+.2f}σ, {pct:+.2f}%)")
print()

# Two-loop prediction
two_loop_pred = run_rge(two_loop_rge, ALPHA_GUT_OBS, M_UNIF, M_Z)
print("-"*78)
print(" TWO-LOOP MSSM RG (Path 4 test):")
print("-"*78)
for i, (pred, tgt, sig) in enumerate(zip(two_loop_pred, pdg, pdg_sig), start=1):
    delta = pred - tgt
    nsig = delta / sig
    pct = delta / tgt * 100
    print(f"   1/α_{i}(M_Z):  predicted = {pred:8.4f},   PDG = {tgt:8.4f},   "
          f"Δ = {delta:+.4f}  ({nsig:+.2f}σ, {pct:+.2f}%)")
print()

# ============================================================
# Two-loop shifts and comparison
# ============================================================
print("-"*78)
print(" TWO-LOOP SHIFTS (= two-loop − one-loop):")
print("-"*78)
for i, (one, two) in enumerate(zip(one_loop_pred, two_loop_pred), start=1):
    shift = two - one
    print(f"   sector {i}: Δ(1/α) = {shift:+.4f}  (sign: {'increases 1/α' if shift > 0 else 'decreases 1/α'})")
print()

# ============================================================
# Sector-specific c extraction at TWO-LOOP precision
# ============================================================
print("-"*78)
print(" SECTOR-SPECIFIC c EXTRACTION (two-loop): what c_i closes each sector?")
print("-"*78)
print()
print(" Method: at two-loop precision, find the uniform-c value c_i that makes")
print(" the two-loop prediction match PDG sector-by-sector. If all c_i ≈ 1/3,")
print(" then sector-specific c is dissolved by two-loop running.")
print()

# To extract c_i at two-loop level: invert α_GUT_unif(c_i) for each sector.
# Two-loop RGE depends non-linearly on initial condition, so we solve numerically.
from scipy.optimize import brentq

def predict_inv_alpha_i_two_loop(c, sector_idx):
    """Predict 1/α_i(M_Z) under TWO-loop running with uniform-c dark correction."""
    alpha_GUT_c = ALPHA_GUT_BARE * (1.0 - c * X_DARK)
    pred = run_rge(two_loop_rge, alpha_GUT_c, M_UNIF, M_Z)
    return pred[sector_idx]

c_extracted = []
for i in range(3):
    target = pdg[i]
    # Solve predict_inv_alpha_i_two_loop(c, i) = target for c
    # Try wide range — c may be negative or > 1
    f = lambda c: predict_inv_alpha_i_two_loop(c, i) - target
    # Probe endpoint signs
    for (lo, hi) in [(-2.0, 1.0), (-5.0, 2.0), (-20.0, 20.0)]:
        try:
            f_lo = f(lo)
            f_hi = f(hi)
            if f_lo * f_hi < 0:
                c_sol = brentq(f, lo, hi)
                c_extracted.append(c_sol)
                break
        except Exception:
            continue
    else:
        c_extracted.append(None)
        # Show endpoint values for diagnostics
        try:
            print(f"   sector {i+1}: extraction failed — no sign change in [-20, 20].")
            print(f"     f(-20)={f(-20):+.3f}, f(0)={f(0):+.3f}, f(1)={f(1):+.3f}, f(20)={f(20):+.3f}")
        except Exception as e:
            print(f"   sector {i+1}: extraction failed — {e}")

for i, c in enumerate(c_extracted, start=1):
    if c is not None:
        delta_from_1_3 = c - 1.0/3.0
        in_range = 0 <= c <= 1
        in_range_str = "" if in_range else "  ⚠ OUTSIDE [0,1]"
        print(f"   sector {i}: c_{i} (two-loop) = {c:+.5f}    "
              f"Δ from 1/3 = {delta_from_1_3:+.5f}{in_range_str}")

print()

# Verdict logic
c1, c2, c3 = c_extracted
print("="*78)
print(" VERDICT")
print("="*78)
print()
print(f"   one-loop:  c_1={0.34281:.4f}, c_2={0.33168:.4f}, c_3={0.24135:.4f}")
fmt = lambda c: f"{c:+.4f}" if c is not None else "FAILED"
print(f"   two-loop:  c_1={fmt(c1)}, c_2={fmt(c2)}, c_3={fmt(c3)}")
print()

# Test: is uniform c = 1/3 consistent at 2σ on all sectors under two-loop?
test_consistent = all(abs(predict_inv_alpha_i_two_loop(1/3, i) - pdg[i]) <= 2 * pdg_sig[i] for i in range(3))

# Test: does c_3 close to within 0.01 of 1/3 under two-loop?
test_c3_closed = c3 is not None and abs(c3 - 1.0/3.0) < 0.01

# Test: does the c_3 vs c_1=c_2 spread reduce significantly?
spread_one_loop = 0.34281 - 0.24135  # = 0.10146
if all(c is not None for c in c_extracted):
    spread_two_loop = max(c_extracted) - min(c_extracted)
    spread_str = f"{spread_two_loop:.4f}"
else:
    spread_two_loop = float('nan')
    spread_str = "N/A (extraction failed)"

if c3 is not None:
    print(f"   c_3 - 1/3 (two-loop): {c3 - 1/3:+.4f}  (target: |Δ| < 0.01 for closure)")
print(f"   spread c_max - c_min: one-loop {spread_one_loop:.4f}  →  two-loop {spread_str}")
print()

if test_c3_closed:
    print(" ✓ TWO-LOOP CLOSES α_3 RESIDUAL with uniform c = 1/3.")
    print("   ⇒ Sector-specific c is ONE-LOOP-RUNNING ARTIFACT.")
    print("   ⇒ Framework's existing uniform c = 1/3 theorem stands.")
    print("   ⇒ W21/W22 obstructions are non-issues (no sector-specific c needed).")
    print()
    print(" RECOMMENDED ACTION: update theorem_alpha_GUT_dark_correction.md §6 to use")
    print(" two-loop MSSM RG instead of one-loop. Add this probe as Type-3 anchor.")
elif test_consistent:
    print(" ◐ TWO-LOOP MARGINALLY CONSISTENT with uniform c = 1/3 at 2σ on all sectors.")
    print("   ⇒ Sector-specific c may be marginal artifact, but the gap is not fully closed.")
    print("   ⇒ Worth investigating threshold corrections (e.g., top-quark threshold at M_t,")
    print("     or QCD hadronic-vacuum-polarization correction at low scales).")
else:
    print(" ✗ TWO-LOOP DOES NOT CLOSE α_3 RESIDUAL with uniform c = 1/3.")
    if c3 is not None:
        print(f"   c_3 = {c3:.4f} (two-loop) — still significantly different from 1/3.")
    print("   ⇒ Sector-specific c is structurally real, not artifact.")
    print("   ⇒ Need to pursue Path 1 (Cl(6) Fock + lattice gauge link variables) or")
    print("     Path 2 (Pati-Salam Killing-form per-bundle).")
    print()
    print(" RECOMMENDED ACTION: Path 1 (Cl(6) Fock gauge action) — but the W22 commutator")
    print(" finding suggests structural obstacles remain.")

print()
print("="*78)
print(" References:")
print("   Martin, S.P. (1997). A Supersymmetry Primer. hep-ph/9709356,")
print("   §6.4 Eq. 6.30 (two-loop gauge β-functions for MSSM).")
print("="*78)
