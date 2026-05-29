#!/usr/bin/env python3
"""
proofs/foundations/b4_a4_t_reversal_bridge_probe.py

PROBE: Does the substrate's time-reversal asymmetry fix the Cl(6) γ_7 sign
       convention?

CONTEXT
-------
Predecessor: an internal working note.
Companion: `b4_a4_dirac_index_probe.py` (per-fiber Dirac index, ind = 0
across BZ — closed negative for spatial-parity bridge).

Spatial-parity bridge (lattice enantiomer srs ↔ srs* → Cl(6) γ_7 sign) was
RULED OUT analytically: the framework's Brauer-Weyl Cl(6) construction has
2 γ-matrices per spatial axis, so any spatial transformation flips an even
number of γ's, leaving γ_7 = γ¹...γ⁶ invariant.

This probe tests an alternative bridge: substrate **time-reversal**
asymmetry. Under T-reversal of fermionic modes (a_i ↔ a_i†):
  γ_{2i-1} = a_i + a_i†     →  invariant
  γ_{2i}   = i(a_i† − a_i)  →  −γ_{2i}
giving γ_7 → (−1)³ γ_7 = −γ_7. So Cl(6) γ_7 IS T-odd by construction.

The substrate IS demonstrably T-asymmetric (p_create = 1/2 ≠ p_destroy = 1/3,
foundation of ε_CP = 1/5 derivation, theorem-grade Row P28).

LOAD-BEARING QUESTION
---------------------
Is there a substrate-natural quantity X such that:
  (1) X is a function of γ_7 and substrate-derived operators (B(P)|_VRam,
      U_{C_3}, etc.) on V_Ram(P);
  (2) X has a definite, non-zero sign in the substrate's "forward-time"
      frame (where arg(h) > 0);
  (3) X flips sign under γ_7 → −γ_7 (i.e., X is γ_7-odd / T-odd).

If yes: γ_7 sign is bridged to substrate time-asymmetry; ADOPTED-B3 (a)
closure candidate.

If no: T-reversal of γ_7 doesn't propagate to a substrate observable;
the (a) sub-question remains genuinely open.

METHODOLOGY
-----------
1. Build B(P)|_VRam (8×8) and γ_7|_VRam (8×8) using the gamma7_chirality.py
   bridge (V_Ram(P) ≅ S as Spin(6) reps via C_3-intertwining isomorphism A).
2. Verify γ_7 is well-defined on V_Ram and squares to I.
3. Compute chirality-graded traces:
     Tr(γ_7 · B^L)  for L = 0, 1, 2, ..., 8
     Tr(γ_7 · B · B†)
     Tr(γ_7 · (B + B†) / 2)
     Tr(γ_7 · (B − B†) / 2i)
   for various polynomials in B and B†.
4. Identify any nonzero such trace.
5. For each nonzero trace, check whether its sign is determined by arg(h) > 0
   (substrate's selected time direction) — i.e., whether the value depends
   on the choice of h-eigenspace orientation in V_Ram.

GATE STATUS
-----------
CAS verification only. Either positive (specific Tr(γ_7 · X) ≠ 0 with
substrate-determined sign) or negative (all such traces vanish or have
arbitrary signs).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges, bloch_hashimoto, build_c3_on_directed_edges,
)

K_P = (0.25, 0.25, 0.25)
TOL = 1e-9


# =====================================================================
# Step 1: Build B(P) and V_Ram(P), restrict B
# =====================================================================

bonds = find_bonds()
directed = build_directed_edges(bonds)
B12 = bloch_hashimoto(K_P, directed)
U12 = build_c3_on_directed_edges(directed)

evals_B, evecs_B = la.eig(B12)
ram_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 2.0) < 1e-6]
assert len(ram_idx) == 8
V_Ram_basis, _ = la.qr(evecs_B[:, ram_idx])
V_Ram_basis = V_Ram_basis[:, :8]

B_VR = V_Ram_basis.conj().T @ B12 @ V_Ram_basis   # 8×8
U_VR = V_Ram_basis.conj().T @ U12 @ V_Ram_basis   # 8×8


# =====================================================================
# Step 2: Build Cl(6) γ-matrices (Brauer-Weyl) and γ_7
# =====================================================================

def build_cl6():
    a_dag = [np.zeros((8, 8), complex) for _ in range(3)]
    for s in range(8):
        bits = [(s >> j) & 1 for j in range(3)]
        for i in range(3):
            if bits[i] == 0:
                ns = s | (1 << i)
                sign = (-1) ** sum(bits[j] for j in range(i))
                a_dag[i][ns, s] = sign
    gammas = []
    for i in range(3):
        a = a_dag[i].conj().T
        gammas.append(a_dag[i] + a)
        gammas.append(1j * (a_dag[i] - a))
    g7 = (-1j) ** 3 * gammas[0] @ gammas[1] @ gammas[2] @ gammas[3] @ gammas[4] @ gammas[5]
    return gammas, g7


gammas, G7_S = build_cl6()


# =====================================================================
# Step 3: Build C_3 on Cl(6) spinor S (Spin(6) lift of body-diagonal)
# =====================================================================
# Reuse the construction from gamma7_chirality.py. The body-diagonal C_3
# in the SO(6) action on Cl(6,0) lifts to a Spin(6) element; the
# corresponding 8×8 unitary on S commutes with γ_7.

from proofs.foundations.gamma7_chirality import build_U_C3_S, c3_isotypic_basis, classify_c3
U_C3_S = build_U_C3_S(directed)


# =====================================================================
# Step 4: Build the C_3-intertwining isomorphism A: S → V_Ram
# =====================================================================

bases_S = c3_isotypic_basis(U_C3_S)
bases_VR = c3_isotypic_basis(U_VR)
A = np.zeros((8, 8), complex)
for label in ['1', 'w', 'w2']:
    A += bases_VR[label] @ bases_S[label].conj().T
assert la.norm(A @ A.conj().T - np.eye(8)) < TOL, "A not unitary"
assert la.norm(A @ U_C3_S - U_VR @ A) < TOL, "A doesn't intertwine C_3"

G7_VR = A @ G7_S @ A.conj().T   # 8×8 chirality on V_Ram


# Sanity: γ_7² = I on V_Ram, and γ_7 splits 4+4
assert la.norm(G7_VR @ G7_VR - np.eye(8)) < TOL
g7_evs = la.eigvalsh(G7_VR)
assert sum(1 for x in g7_evs if x > 0.5) == 4
assert sum(1 for x in g7_evs if x < -0.5) == 4


# =====================================================================
# Step 5: Compute candidate T-odd substrate traces
# =====================================================================

print("=" * 72)
print(" b4_a4_t_reversal_bridge_probe — chirality-graded substrate traces")
print("=" * 72)
print()
print(f"  B(P)|_VRam Ramanujan eigenvalues:")
for ev in sorted(la.eigvals(B_VR), key=lambda z: (np.angle(z), z.real)):
    print(f"    {ev:+.4f}    arg = {np.degrees(np.angle(ev)):+.2f}°    |ev|² = {abs(ev)**2:.4f}")
print()
print(f"  γ_7|_VRam diagonalized:  +1 mult 4,  -1 mult 4")
print(f"  ||[γ_7|_VRam, U_{{C_3}}|_VRam]|| = {la.norm(G7_VR @ U_VR - U_VR @ G7_VR):.2e}")
print()


def sup_trace(M):
    """The supertrace Tr(γ_7 · M) — γ_7-graded trace on V_Ram(P)."""
    return np.trace(G7_VR @ M)


# Polynomial probes
print("=" * 72)
print(" Chirality-graded traces Tr(γ_7 · M) for various M:")
print("=" * 72)
print()

I8 = np.eye(8, dtype=complex)
Bd = B_VR.conj().T
candidates = {
    "I":              I8,
    "B":              B_VR,
    "B†":             Bd,
    "B + B†":         B_VR + Bd,
    "(B - B†) / 2i":  (B_VR - Bd) / (2j),
    "B B†":           B_VR @ Bd,
    "B²":             B_VR @ B_VR,
    "(B†)²":          Bd @ Bd,
    "B² + (B†)²":     B_VR @ B_VR + Bd @ Bd,
    "B² - (B†)²":     B_VR @ B_VR - Bd @ Bd,
    "B³":             B_VR @ B_VR @ B_VR,
    "B B† B":         B_VR @ Bd @ B_VR,
    "B† B B†":        Bd @ B_VR @ Bd,
    "B^4":            la.matrix_power(B_VR, 4),
    "B^5":            la.matrix_power(B_VR, 5),
    "B^6":            la.matrix_power(B_VR, 6),
    "B^7":            la.matrix_power(B_VR, 7),
    "B^8":            la.matrix_power(B_VR, 8),
    # C_3-coupled
    "U_{C_3}":        U_VR,
    "B · U_{C_3}":    B_VR @ U_VR,
    "B² · U_{C_3}":   B_VR @ B_VR @ U_VR,
}

nonzero_found = []
for name, M in candidates.items():
    val = sup_trace(M)
    is_zero = abs(val) < 1e-9
    print(f"  Tr(γ_7 · {name:>14}) = {val.real:+.6f}  +  {val.imag:+.6f} i"
          f"   {'(≈ 0)' if is_zero else '(NONZERO)'}")
    if not is_zero:
        nonzero_found.append((name, val))

# =====================================================================
# Step 6: Verdict
# =====================================================================

print()
print("=" * 72)
print(" Verdict")
print("=" * 72)
print()

if nonzero_found:
    print(f"  ✓ Found {len(nonzero_found)} nonzero chirality-graded substrate trace(s).")
    print(f"    Each is γ_7-odd: under γ_7 → −γ_7 (substrate T-reversal), they flip sign.")
    print(f"    These are candidate substrate-natural T-odd quantities.")
    print()
    print(f"    For ADOPTED-B3 (a) closure, the load-bearing question now is:")
    print(f"    does AT LEAST ONE of these traces have a sign that's determined")
    print(f"    by the substrate's forward-time arrow (arg(h) > 0), independent")
    print(f"    of the (Z/2)^3 labeling-convention freedom in B3?")
    print()
    print(f"    Candidates above with substantial magnitude:")
    for name, val in sorted(nonzero_found, key=lambda x: -abs(x[1]))[:5]:
        print(f"      Tr(γ_7 · {name:>14}) = {val.real:+.6f} + {val.imag:+.6f} i,  "
              f"|.| = {abs(val):.4f}")
else:
    print(f"  ✗ All tested chirality-graded substrate traces are zero.")
    print(f"    No substrate-natural T-odd quantity exists at this layer:")
    print(f"    γ_7|_VRam is 'trace-orthogonal' to all polynomials of B(P)|_VRam.")
    print(f"    The substrate's time-asymmetry doesn't propagate to fix γ_7 sign")
    print(f"    via this mechanism.")
    print()
    print(f"    ADOPTED-B3 (a) parity convention remains genuinely open.")

print()
print("=" * 72)
print(" OK: b4_a4_t_reversal_bridge_probe complete.")
print("=" * 72)
