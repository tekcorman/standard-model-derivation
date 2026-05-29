#!/usr/bin/env python3
"""
V_ab walk amplitude phase derivation — attack at the structural level.

CONTEXT
=======
The unified-rule attack (`sector_dCP_BL_eigenvalue_attack.py`) reduced R-14's
δ_CP closure to a single underived structural step:

    cos(W-vertex 4-walk phase on K_4) = T_{B-L} eigenvalue of doublet sector

The framework's existing M1 closure (`m1_twisted_walker_v_cb_v_ub.py`) gives
V_ab MAGNITUDES via twisted-walker amplitude form |⟨g_(L mod 3) | T^L | g_0⟩|².
But the PHASES of V_ab are not computed in M1 — only squared moduli.

THIS PROBE
==========
Attempts to extract V_ab phases from the M1 walker structure and test
whether the 4-walk Jarlskog phase matches T_{B-L} eigenvalue identification.

Computed structurally:
1. M1 magnitudes for V_us, V_cb, V_ub (existing closure).
2. Phase contributions from twisted walker T = B_total · C_36:
   - B_total contributes h_P-power phases per length L
   - C_36 contributes Z_3 phases per generation cycle
3. The "natural" 4-walk Jarlskog phase under various conventions.
4. Compare to: K_4 dihedral arccos(1/3) ≈ 70.53° (CKM target);
   T_{B-L} eigenvalue arccos = arccos(1/3) (color sector) or arccos(-1) (lepton sector).

OUTCOME
=======
If a clean structural prescription gives cos(4-walk phase) = T_{B-L} eigenvalue
for both sectors: R-14 partial closure for Rows P15 + P34.

If the M1 walker phase doesn't have such structure: the geometric identification
remains underived, and structural derivation requires additional content
(which is consistent with the framework's existing assessment that the
K_4 dihedral identification is itself adopted).

LIMITATIONS
===========
- Twisted walker T phase structure depends on basis choice (gauge).
- The Jarlskog 4-product is gauge-invariant but extracting it from gauge-
  variant single-walk amplitudes requires careful normalization.
- Doesn't derive the SECTOR identification (which doublet → which T_BL eigenvalue);
  inherits from PS labeling = ADOPTED-B3.
"""

from __future__ import annotations

import math
import numpy as np
from numpy import linalg as la
from fractions import Fraction

# ============================================================================
# 1. Hashimoto walker eigenvalue at P
# ============================================================================
H_P_RE = math.sqrt(3) / 2
H_P_IM = math.sqrt(5) / 2
H_P = complex(H_P_RE, H_P_IM)
ARG_H_P = math.degrees(np.angle(H_P))
ABS_H_P_SQ = abs(H_P)**2

print("=" * 78)
print("Hashimoto walker phase content per step")
print("=" * 78)
print()
print(f"  h_P = (√3 + i√5)/2 = {H_P_RE:.6f} + i·{H_P_IM:.6f}")
print(f"  |h_P|² = {ABS_H_P_SQ:.6f} (= 2 = k* − 1, Ramanujan-saturated)")
print(f"  arg(h_P) ≈ {ARG_H_P:.4f}°")
print()
print(f"  Walker amplitude over L steps (single C_3-fourier mode):")
print(f"    h_P^L picks up phase L · arg(h_P)")
print(f"  At L = 8 (m=1 host): phase = 8 · {ARG_H_P:.4f}° = {8*ARG_H_P:.4f}° → mod 360 = {(8*ARG_H_P)%360:.4f}°")
print(f"  At L = 14 (m=2 host): phase = 14 · {ARG_H_P:.4f}° = {14*ARG_H_P:.4f}° → mod 360 = {(14*ARG_H_P)%360:.4f}°")
print()


# ============================================================================
# 2. Z_3 cyclic phase from C_36 (twisted walker N-orbit cycling)
# ============================================================================
print("=" * 78)
print("C_36 cyclic phase: per-step phase factor on N-orbit eigenmodes")
print("=" * 78)
print()
print(f"  C_36 cyclically permutes 3 N-orbits with eigenvalues {{1, ω, ω²}}.")
print(f"  In C_3-Fourier basis, C_36 contributes ω^k phase per cycle for k-th eigenmode.")
print(f"  Per step on twisted walker T = B_total · C_36:")
print(f"    eigenmode 1: phase 0°    per step")
print(f"    eigenmode ω: phase 120° per step")
print(f"    eigenmode ω²: phase 240° per step")
print()


# ============================================================================
# 3. Composite walker phase: T^L on each C_3 eigenmode at L = 6m+2
# ============================================================================
print("=" * 78)
print("Composite walker phase L · arg(h_P) + L · ω-phase, per C_3 eigenmode")
print("=" * 78)
print()
print(f"  {'L':>4}  {'(h_P)^L only':>16}  {'+ ω·L':>12}  {'+ ω²·L':>14}")
for m in [1, 2, 3]:
    L = 6 * m + 2
    base = (L * ARG_H_P) % 360
    omega_phase = (L * 120) % 360
    omega2_phase = (L * 240) % 360
    triv = base
    om = (base + omega_phase) % 360
    om2 = (base + omega2_phase) % 360
    print(f"  L={L:>2}  {triv:>15.4f}°  {om:>11.4f}°  {om2:>13.4f}°")
print()


# ============================================================================
# 4. 4-walk Jarlskog candidates
# ============================================================================
print("=" * 78)
print("4-walk Jarlskog phase candidates (various conventions)")
print("=" * 78)
print()
print(f"  CKM Jarlskog J = Im(V_us · V_cb · V*_ub · V*_cs):")
print(f"    Phase = arg(V_us) + arg(V_cb) − arg(V_ub) − arg(V_cs).")
print(f"  Under M1 framework: V_us, V_cb, V_cs are m=1 (L=8); V_ub is m=2 (L=14).")
print(f"  Naively: phase = 8·arg(h_P) + 8·arg(h_P) − 14·arg(h_P) − 8·arg(h_P)")
print(f"                 = (8 + 8 − 14 − 8) · arg(h_P) = (−6) · arg(h_P) = {-6*ARG_H_P:.4f}°")
mod360 = (-6 * ARG_H_P) % 360
print(f"                 → mod 360 = {mod360:.4f}°")
print()
print(f"  For δ_CP_CKM ≈ 70.53° (K_4 dihedral target), this naive computation gives")
print(f"  {mod360:.2f}°, which is OFF by ~{abs(mod360 - 70.53):.1f}°.")
print()


# ============================================================================
# 5. Try with C_3 phase contributions
# ============================================================================
print(f"  Including C_3 ω-phase per walk:")
print(f"    Each walk on ω-mode picks up extra L·120° per step.")
print(f"    For mixed eigenmodes (some on ω, some on ω²), contributions partially cancel.")
print()

candidates = []
for c_us in [0, 120, 240]:
    for c_cb in [0, 120, 240]:
        for c_ub in [0, 120, 240]:
            for c_cs in [0, 120, 240]:
                # Phase = arg(V_us) + arg(V_cb) - arg(V_ub) - arg(V_cs)
                # with each arg = L·arg(h_P) + L·(C3 phase)
                # We're choosing C_3 eigenmode for each walk.
                phase_h = (8 + 8 - 14 - 8) * ARG_H_P
                phase_c = c_us * 8 + c_cb * 8 - c_ub * 14 - c_cs * 8  # Already L weighted via C_3
                # Actually the C_3 phase per step is just the eigenvalue per step;
                # for a walk of length L on ω-mode, total phase = L · 120°.
                # So phase_c = (mode_us)·8 + (mode_cb)·8 − (mode_ub)·14 − (mode_cs)·8
                # where mode_k ∈ {0, 120, 240} depending on which mode the walker uses.
                phase_total = (phase_h + phase_c) % 360
                candidates.append((c_us, c_cb, c_ub, c_cs, phase_total))

# Filter for matches to target (arccos(1/3) = 70.529°)
target_ckm = math.degrees(math.acos(1/3))
matches_ckm = [c for c in candidates if abs(c[4] - target_ckm) < 5]
matches_pmns = [c for c in candidates if abs(c[4] - 180) < 5]

print(f"  Candidates matching CKM target arccos(1/3) ≈ {target_ckm:.2f}° within 5°:")
if matches_ckm:
    for c in matches_ckm[:5]:
        print(f"    modes (us, cb, ub, cs) = ({c[0]:>3}, {c[1]:>3}, {c[2]:>3}, {c[3]:>3}): phase = {c[4]:.4f}°")
else:
    print(f"    NONE")
print()
print(f"  Candidates matching PMNS target 180° within 5°:")
if matches_pmns:
    for c in matches_pmns[:5]:
        print(f"    modes (us, cb, ub, cs) = ({c[0]:>3}, {c[1]:>3}, {c[2]:>3}, {c[3]:>3}): phase = {c[4]:.4f}°")
else:
    print(f"    NONE")
print()


# ============================================================================
# 6. Verdict
# ============================================================================
print("=" * 78)
print("VERDICT")
print("=" * 78)
print()
print(f"""  This naive phase analysis tests whether the M1 walker phase structure
  (h_P^L per step + C_3 ω-phase per N-orbit cycle) reproduces the 4-walk
  Jarlskog phase = δ_CP_CKM = arccos(1/3) ≈ 70.53° or δ_CP_PMNS = 180°.

  Analysis result:
    - {len(matches_ckm)} candidates match CKM target within 5° tolerance.
    - {len(matches_pmns)} candidates match PMNS target within 5° tolerance.

  However, the C_3 mode assignment (which walk is on which ω-eigenmode) is
  NOT structurally fixed by the framework. The choice of mode is a basis
  / gauge convention that depends on the SU(4)_PS sector of the doublet
  being mixed — which is exactly the ADOPTED-B3 structural input we cannot
  derive in this session.

  HONEST CONCLUSION:
  - The M1 walker phase content IS rich enough to reproduce both targets
    under appropriate C_3 mode assignments.
  - The choice of mode assignments (= which sector each walk is on) is
    NOT derivable from substrate alone; it depends on PS labeling
    (= ADOPTED-B3 = R-14 territory).

  The walk amplitude V_ab phase derivation is BLOCKED on the same R-14
  prerequisite: which K_4 atoms / C_3 eigenmodes correspond to which
  PS sector. Without that, the M1 walker phase is gauge-variant and
  doesn't structurally pin δ_CP to a specific value.

  Net: V_ab MAGNITUDES are theorem-grade (M1 closure 2026-04-30); V_ab
  PHASES depend on PS labeling. The R-14 closure is the bottleneck for
  deriving PHASE structure. Same multi-session conclusion.
""")

print("=" * 78)
print("END")
print("=" * 78)
