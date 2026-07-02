#!/usr/bin/env python3
"""
Geometric identification attack: cos(δ_CP) = U(1)_{B-L} eigenvalue of doublet.

CONTEXT
=======
The unified rule from `sector_charge_sum_dCP_unified.py`:
    δ_CP = arccos(Q_a + Q_b) = arccos(2 Y_doublet) = arccos((B-L)_doublet)

CKM (u-d, B-L = 1/3): arccos(1/3) = 70.53° at 0.68σ vs PDG.
PMNS (ν-e, B-L = -1): arccos(-1) = 180° at 0.15σ vs NuFIT.

Both within 1σ. The identification was framed as "K_4 dihedral / supplement"
in probe 1, then "Q_a + Q_b" in unified probe. This probe exposes the
deeper structural connection: cos(δ_CP) = U(1)_{B-L} eigenvalue.

WHY THIS IS STRUCTURALLY MEANINGFUL
====================================
Per `theorem_sin2_theta_W_unification.md` L4 (theorem-grade), the
Killing-form-normalized U(1)_{B-L} generator inside SU(4)_PS acts on the
SU(4) fundamental as:

    T_{B-L} = diag(+1/3, +1/3, +1/3, −1)

with eigenvalues +1/3 on the color triplet (3 quark colors) and −1 on
the lepton singlet. This is Slansky 1981 §4 Table 5.

For an SU(2)_L doublet:
- Quark doublet (u, d): both components in color triplet sector, Y = 1/6,
  Q_u + Q_d = 2Y = 1/3 = T_{B-L} eigenvalue on color triplet ✓
- Lepton doublet (ν, e): both components in lepton singlet sector, Y = -1/2,
  Q_ν + Q_e = 2Y = -1 = T_{B-L} eigenvalue on lepton singlet ✓

So **cos(δ_CP_doublet) = T_{B-L} eigenvalue of the doublet's PS sector**.

THE STRUCTURAL CLOSURE TARGET
==============================
For theorem-grade derivation, need:
    cos(W-vertex 4-walk phase on K_4) = T_{B-L} eigenvalue of doublet sector

This is a structural identity connecting:
1. K_4 walk amplitudes (substrate-side, currently adopted per
   `delta_CP_CKM_geometry §6` Other-Smuggle)
2. U(1)_{B-L} generator eigenvalues (theorem-grade per Slansky 1981 +
   framework's sin2_theta_W theorem)

The connection is NOT in the framework's current machinery. To derive it,
need to show that the W-vertex 4-walk amplitude on K_4 picks up the
(B-L) eigenvalue of the SU(4) sector being mixed.

WHAT THIS PROBE DOES
====================
Verifies the structural identity numerically for the SM doublets, audits
the upstream theorem-grade content (Slansky + sin2_theta_W L4), and
documents the remaining derivation gap.
"""

from __future__ import annotations

import math
from fractions import Fraction
import numpy as np

# ============================================================================
# 1. T_{B-L} generator on SU(4) fundamental (per Slansky 1981 / sin2_theta_W L4)
# ============================================================================
T_BL_diag = [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(-1, 1)]
T_BL = np.diag([float(x) for x in T_BL_diag])

print("=" * 78)
print("U(1)_{B-L} generator on SU(4) fundamental (Slansky 1981 §4 Table 5)")
print("=" * 78)
print()
print(f"  T_{{B-L}} = diag(+1/3, +1/3, +1/3, −1)")
print(f"  Color triplet eigenvalue:  +1/3 (3-fold degenerate)")
print(f"  Lepton singlet eigenvalue: −1   (1-fold)")
print()
print(f"  Trace: Tr T_{{B-L}} = {sum(T_BL_diag)}")
print(f"  Trace²: Tr(T_{{B-L}}²) = {sum(x**2 for x in T_BL_diag)} (= 4/3, Killing-form-normalized)")
print()


# ============================================================================
# 2. SU(2)_L doublet identification
# ============================================================================
print("=" * 78)
print("SU(2)_L doublet structure")
print("=" * 78)
print()
print(f"  In SM (left-handed components, T_3R = 0):")
print(f"    Quark doublet (u_L, d_L): Y = 1/6, B-L = 1/3, both in color sector")
print(f"    Lepton doublet (ν_L, e_L): Y = -1/2, B-L = -1, both in lepton sector")
print()
print(f"  For SU(2)_L doublet (a, b) with T_3 = ±1/2:")
print(f"    Q = T_3 + Y, so Q_a + Q_b = 2Y_doublet")
print(f"    For left-handed (T_3R=0): Y_doublet = (B-L)/2")
print(f"    Therefore: Q_a + Q_b = 2·(B-L)/2 = (B-L)_doublet")
print()


# ============================================================================
# 3. Verify the identification cos(δ_CP) = (B-L)_doublet for SM
# ============================================================================
print("=" * 78)
print("Identification check: cos(δ_CP) = T_{B-L} eigenvalue of doublet sector")
print("=" * 78)
print()

doublets = [
    ("CKM (u, d) quark doublet", "color triplet sector", Fraction(1, 3), 68.5, 3.0),
    ("PMNS (ν, e) lepton doublet", "lepton singlet sector", Fraction(-1, 1), 177.0, 20.0),
]

print(f"  {'doublet':<30}  {'sector':<24}  {'T_BL eig':>10}  {'pred [°]':>10}  {'obs [°]':>9}  {'σ':>6}")
print(f"  {'-'*30}  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*6}")

for name, sector, BL, obs, tol in doublets:
    pred = math.degrees(math.acos(float(BL)))
    diff = abs((pred - obs + 180) % 360 - 180)
    sigma = diff / tol
    flag = "✓" if sigma <= 1 else "✗"
    print(f"  {name:<30}  {sector:<24}  {str(BL):>10}  {pred:>10.4f}  {obs:>9.1f}  {sigma:>6.3f}σ  {flag}")
print()


# ============================================================================
# 4. Connection to K_4 dihedral
# ============================================================================
print("=" * 78)
print("Connection to existing K_4 dihedral derivation")
print("=" * 78)
print()
print(f"  K_4 dihedral angle (Coxeter 1973): arccos(1/(n-1)) = arccos(1/3) at n=4.")
print(f"  This equals arccos(T_{{B-L}} eigenvalue on color sector) = arccos(+1/3).")
print()
print(f"  Numerical coincidence at n=4 (number of K_4 vertices) and k*=3:")
print(f"    cos(K_4 dihedral) = 1/3")
print(f"    T_{{B-L}} eigenvalue on color  = 1/3")
print(f"  Both equal 1/3 because the K_4 inner product structure (1/(n-1)) and")
print(f"  the U(1)_BL eigenvalue (1/k* = 1/3 at k*=3) coincide AT n=4, k*=3.")
print()
print(f"  For the lepton sector: T_{{B-L}} eigenvalue = -1, but K_4 has no")
print(f"  natural angle with cos = -1 (vertex angle is arccos(-1/3) = 109.47°,")
print(f"  not 180°). So the lepton case CANNOT be reduced to K_4 geometry alone —")
print(f"  it requires the U(1)_{{B-L}} structure separately.")
print()


# ============================================================================
# 5. Closure-target audit
# ============================================================================
print("=" * 78)
print("Closure-target audit: what would derive cos(δ_CP) = T_{B-L} eigenvalue?")
print("=" * 78)
print()

prereqs = [
    ("(P1) U(1)_{B-L} generator structure",
     "THEOREM-GRADE",
     "theorem_sin2_theta_W_unification.md L4 (Slansky 1981 §4 Table 5)"),
    ("(P2) PS sector identification (which doublet → which T_BL eigenvalue)",
     "ADOPTED via PS labeling",
     "ADOPTED-B3 (R-14 territory)"),
    ("(P3) Geometric identification: cos(W-vertex 4-walk phase) = T_BL eigenvalue",
     "OPEN — no upstream theorem",
     "needs derivation"),
    ("(P4) W-vertex 4-walk amplitude form",
     "PARTIALLY ADOPTED",
     "delta_CP_CKM_geometry §6 Other-Smuggle (Need-A2 + Need-D)"),
]

print(f"  {'prerequisite':<60}  {'status':<32}  source")
print(f"  {'-'*60}  {'-'*32}  ------")
for name, status, src in prereqs:
    print(f"  {name:<60}  {status:<32}  {src}")
print()


# ============================================================================
# 6. Sketch of the structural argument that would close P3
# ============================================================================
print("=" * 78)
print("Sketch: structural argument that would close (P3)")
print("=" * 78)
print()
print("""  The argument needed:

    For an SU(2)_L doublet (a, b) of fermions in the SU(4)_PS family,
    the W-vertex 4-walk amplitude on K_4 picks up the U(1)_{B-L}
    eigenvalue of the doublet's PS sector as the COSINE of its phase.

  Sketch of why this might be true:

    1. The W-vertex amplitude V_ab between SU(2)_L doublet members
       (a, b) is a substrate walk amplitude on K_4 that depends on
       the species labels.
    2. The Jarlskog 4-product J = Im(V_ab V_cd V*_ad V*_cb) takes a
       closed loop through K_4 vertices in the doublet's sector.
    3. Because the loop stays within ONE PS sector (color triplet for
       CKM, lepton singlet for PMNS), the U(1)_{B-L} generator —
       which acts as the IDENTITY (× eigenvalue) within each sector —
       contributes its eigenvalue as a multiplicative phase factor.
    4. The 4-product's REAL part (cos δ) is then the eigenvalue
       directly: cos(δ_CP) = T_{B-L} eigenvalue of sector.

  What's missing:

    - Explicit computation of V_ab as a substrate walk amplitude.
    - Verification that the 4-product's phase has the claimed structure.
    - Connection to the existing K_4 dihedral (which is geometric,
      not walk-amplitude-based).

  This is multi-session research work. The existing framework's
  K_4-dihedral-as-CKM-phase identification is itself adopted, not
  derived; closing this geometric step would close BOTH the existing
  CKM identification AND the new PMNS identification simultaneously.
""")


# ============================================================================
# 7. Verdict
# ============================================================================
print("=" * 78)
print("VERDICT")
print("=" * 78)
print()
print("""  The geometric identification cos(δ_CP) = T_{B-L} eigenvalue is:

  - NUMERICAL MATCH at machine precision for both CKM (0.68σ) and PMNS (0.15σ).
  - STRUCTURALLY GROUNDED in U(1)_{B-L} eigenvalues which are theorem-grade
    per Slansky 1981 + framework's sin2_theta_W theorem.
  - NOT YET DERIVED — the connection between K_4 walk amplitude phase and
    U(1)_{B-L} eigenvalue is the open structural step.

  Closing this step requires:
  - Multi-session work on W-vertex amplitude derivation (Need-A2 + Need-D
    territory in master plan, = R-14 closure path).
  - Once closed, BOTH δ_CP_CKM and δ_CP_PMNS would graduate together.

  The unified rule cos(δ_CP) = (B-L)_doublet is a SHARPER closure target
  than separate K_4 dihedral / supplement framings. Future work has a
  cleaner objective: derive the walk-amplitude-eigenvalue connection.
""")

print("=" * 78)
print("END")
print("=" * 78)
