#!/usr/bin/env python3
"""
proofs/foundations/b4_color_vram_gamma_commutant_probe.py

PROBE: Does V_Ram(Γ) host a hidden SU(3) action commuting with B(Γ) and the
       Γ-point stabilizer (full octahedral group O)?

CONTEXT
-------
Companion to `b4_color_vram_p_commutant_probe.py`, which refuted V_Ram(P)
as the SU(3)_color seed for B4 Route (i) M1.

At Γ the point-group stabilizer is the full octahedral group O (order 24,
not just T = 12 as at P). O has irreps (A_1, A_2, E, T_1, T_2) of dims
(1, 1, 2, 3, 3).

Restriction to body-diagonal C_3:
    A_1 ↓ C_3 = trivial      A_2 ↓ C_3 = trivial      E ↓ C_3 = ω ⊕ ω̄
    T_1 ↓ C_3 = 1 ⊕ ω ⊕ ω̄    T_2 ↓ C_3 = 1 ⊕ ω ⊕ ω̄

V_Ram(Γ) is 6-dim (six Ramanujan-saturated eigenvalues μ = (-1 ± i√7)/2,
each with multiplicity 3, from the three-fold-degenerate λ = -1 in
A(Γ) ≃ adjacency of K_4).

The decompositions of a 6-dim space under O that ADMIT SU(3) in the
commutant are sharply restricted:

  decomposition          | commutant in M_6    | room for SU(3)?
  -----------------------|---------------------|-----------------
  3·E                    | M_3                 | YES (su(3) ⊂ u(3) = M_3)
  6·A_1   or  6·A_2      | M_6                 | YES algebraically, NOT canonical
  2·T_1   or  2·T_2      | M_2                 | NO (only su(2))
  1·T_1 + 1·T_2          | M_1 ⊕ M_1           | NO
  4·A + 1·E, 2·A + 2·E,  | smaller             | NO (no 3-dim multiplicity)
  etc.

The C_3-isotypic signature (m_1, m_ω, m_ω̄) distinguishes them sharply:
  3·E         ⇒  (0, 3, 3)
  6·A_i       ⇒  (6, 0, 0)
  2·T_i       ⇒  (2, 2, 2)
  T_1 + T_2   ⇒  (2, 2, 2)
  others      ⇒  various

GATE STATUS
-----------
CAS verification only. Determines whether B4 Route (i) M1 has a candidate
8-dim quadratic space (or here, 6-dim) at Γ via the same commutant test.

Run with:
    PYTHONPATH=. python3 proofs/foundations/b4_color_vram_gamma_commutant_probe.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
    commutator_norm,
)


# =====================================================================
# Step 0: build B(Γ), U_{C_3}, and V_Ram(Γ)
# =====================================================================

K_GAMMA = (0.0, 0.0, 0.0)
RAM_MOD_SQ = 2.0
TOL = 1e-8


def extract_vram(B, expected_dim=None):
    evals, evecs = la.eig(B)
    ram_idx = [i for i, ev in enumerate(evals) if abs(abs(ev) ** 2 - RAM_MOD_SQ) < 1e-6]
    if expected_dim is not None:
        assert len(ram_idx) == expected_dim, \
            f"Expected {expected_dim} Ramanujan eigenvalues, got {len(ram_idx)}"
    V_raw = evecs[:, ram_idx]
    V, _ = la.qr(V_raw)
    V = V[:, :len(ram_idx)]
    return V, np.array([evals[i] for i in ram_idx])


def restrict(M, V):
    return V.conj().T @ M @ V


bonds = find_bonds()
directed = build_directed_edges(bonds)
B12 = bloch_hashimoto(K_GAMMA, directed)
U12 = build_c3_on_directed_edges(directed)

V_Ram, ram_eigs = extract_vram(B12, expected_dim=6)
B = restrict(B12, V_Ram)
U = restrict(U12, V_Ram)

assert la.norm(U.conj().T @ U - np.eye(6)) < TOL
assert la.norm(la.matrix_power(U, 3) - np.eye(6)) < TOL
assert commutator_norm(B, U) < 1e-6, "B(Γ) and U_{C_3} do not commute"

print("=" * 72)
print("Step 0 — V_Ram(Γ), B|_VRam, U_{C_3}|_VRam constructed")
print("=" * 72)
print(f"  V_Ram(Γ) dim = {V_Ram.shape[1]}")
print(f"  Ramanujan eigenvalues of B(Γ)|_VRam:")
for lam in sorted(ram_eigs, key=lambda z: (np.angle(z), z.real)):
    print(f"    {lam:+.6f}   |lam|^2 = {abs(lam)**2:.4f}")
print(f"  ||[B|_VRam, U|_VRam]|| = {commutator_norm(B, U):.2e}")


# =====================================================================
# Step 1: C_3-isotypic decomposition of V_Ram(Γ)
# =====================================================================

def c3_isotypic(U_n):
    n = U_n.shape[0]
    om = omega3
    I_n = np.eye(n)
    P_1 = (I_n + U_n + U_n @ U_n) / 3
    P_w = (I_n + np.conj(om) * U_n + np.conj(om) ** 2 * (U_n @ U_n)) / 3
    P_w2 = (I_n + np.conj(om) ** 2 * U_n + np.conj(om) * (U_n @ U_n)) / 3
    return P_1, P_w, P_w2


P_1, P_w, P_w2 = c3_isotypic(U)
m_1 = int(round(np.real(np.trace(P_1))))
m_w = int(round(np.real(np.trace(P_w))))
m_w2 = int(round(np.real(np.trace(P_w2))))

print()
print("=" * 72)
print("Step 1 — C_3 isotypic decomposition of V_Ram(Γ)")
print("=" * 72)
print(f"  (m_1, m_omega, m_omega^2) = ({m_1}, {m_w}, {m_w2})")
print()
print("  Comparison vs structurally-allowed O-decompositions of V_Ram(Γ):")
print(f"    (0, 3, 3)  ⇒  3·E         (commutant = M_3, hosts u(3) ⊃ su(3))")
print(f"    (2, 2, 2)  ⇒  2·T_1 or 2·T_2 or T_1+T_2 (commutant has only M_2)")
print(f"    (6, 0, 0)  ⇒  6·A_1 or 6·A_2  (commutant = M_6, su(3) not canonical)")
print(f"    other      ⇒  mixed irreps, no 3-dim multiplicity")


# =====================================================================
# Step 2: joint commutant of {B|_VRam, U|_VRam} in M_6(C)
# =====================================================================

def joint_commutant(ops, dim, tol=1e-7):
    n = dim
    eqs = []
    I_n = np.eye(n)
    for X in ops:
        eqs.append(np.kron(I_n, X) - np.kron(X.T, I_n))
    M = np.vstack(eqs)
    _, s, Vh = la.svd(M)
    rank = int((s > tol * (s[0] if s[0] > 0 else 1.0)).sum())
    null = Vh[rank:].conj().T
    dim_null = null.shape[1]
    basis = [null[:, k].reshape(n, n) for k in range(dim_null)]
    return basis, dim_null


comm_basis, dim_comm = joint_commutant([B, U], 6)
print()
print("=" * 72)
print("Step 2 — joint commutant of {B|_VRam, U|_VRam} in M_6(C)")
print("=" * 72)
print(f"  dim(commutant) = {dim_comm}")


# =====================================================================
# Step 3: joint (B, U) eigenvalue structure
# =====================================================================

def joint_block_structure(B, U, dim):
    om = omega3

    evB, vecB = la.eig(B)
    B_groups = {}
    used = [False] * len(evB)
    for i in range(len(evB)):
        if used[i]:
            continue
        key = None
        for k in B_groups:
            if abs(evB[i] - k) < 1e-6:
                key = k
                break
        if key is None:
            key = complex(evB[i])
            B_groups[key] = []
        B_groups[key].append(i)
        used[i] = True
    B_keys = sorted(B_groups.keys(), key=lambda z: (round(np.angle(z), 4), round(z.real, 4)))

    print(f"  B(Γ)|_VRam eigenvalue spectrum:")
    for k in B_keys:
        idx = B_groups[k]
        Q, _ = la.qr(vecB[:, idx])
        Q = Q[:, :len(idx)]
        U_block = Q.conj().T @ U @ Q
        ev_block = la.eigvals(U_block)
        labels = []
        for ev in ev_block:
            if abs(ev - 1) < 0.1:
                labels.append("1")
            elif abs(ev - om) < 0.1:
                labels.append("ω")
            elif abs(ev - om ** 2) < 0.1:
                labels.append("ω̄")
            else:
                labels.append(f"?{ev:.3f}")
        labels.sort()
        print(f"    λ = {k:+.4f}  (mult {len(idx)})   U-content: {{{', '.join(labels)}}}")

    U_eigs = [1.0 + 0j, om, om ** 2]
    U_labels = ["1", "ω", "ω̄"]
    m = {}
    for k in B_keys:
        idx = B_groups[k]
        Q, _ = la.qr(vecB[:, idx])
        Q = Q[:, :len(idx)]
        U_block = Q.conj().T @ U @ Q
        ev_block = la.eigvals(U_block)
        for j, U_ev in enumerate(U_eigs):
            count = sum(1 for ev in ev_block if abs(ev - U_ev) < 0.1)
            m[(k, U_labels[j])] = count

    return B_keys, U_labels, m


print()
print("=" * 72)
print("Step 3 — joint (B, U) structure")
print("=" * 72)
B_keys, U_labels, m_joint = joint_block_structure(B, U, 6)

print()
print(f"  joint multiplicity matrix m_{{ij}} (B-row × U-col):")
header = "     λ\\U " + "   ".join(f"{l:>4}" for l in U_labels) + "    sum"
print(header)
total = 0
for k in B_keys:
    row = [m_joint[(k, l)] for l in U_labels]
    s = sum(row)
    total += s
    row_str = "  ".join(f"{v:4d}" for v in row)
    print(f"  λ={k:+.3f}   {row_str}   {s:4d}")
print(f"  total {total:>30d}")

dim_check = sum(v ** 2 for v in m_joint.values())
print(f"\n  Σ m_{{ij}}^2 = {dim_check}    "
      f"(matches dim(commutant) = {dim_comm}? "
      f"{'✓' if dim_check == dim_comm else '✗'})")


# =====================================================================
# Step 4: verdict
# =====================================================================

print()
print("=" * 72)
print("Step 4 — verdict")
print("=" * 72)
print()

iso = (m_1, m_w, m_w2)

if iso == (0, 3, 3):
    print(f"  ✓ C_3-isotypic = (0, 3, 3) — MATCHES the 3·E pattern.")
    print(f"  V_Ram(Γ) decomposes under O as 3 copies of the 2-dim irrep E.")
    print(f"  Commutant of O (without B): M_3 (acting on E-multiplicity space).")
    print(f"  Joint commutant of {{B(Γ), U_{{C_3}}}}: dim = {dim_comm}.")
    print()
    if dim_comm >= 9:
        print(f"  ✓ B(Γ) does NOT refine the 3-dim multiplicity space — full M_3 ≅ u(3)")
        print(f"    is preserved. SU(3) sits as the traceless part of M_3.")
        print()
        print(f"  >>> POSITIVE: V_Ram(Γ) HOSTS A SU(3) ACTION commuting with the")
        print(f"      substrate's natural operators.")
    else:
        print(f"  ✗ B(Γ) refines the multiplicity space, dim_comm = {dim_comm} < 9.")
        print(f"    Need to check whether su(3) survives the B-refinement.")
elif iso == (2, 2, 2):
    print(f"  C_3-isotypic = (2, 2, 2) — consistent with 2·T_1, 2·T_2, or T_1+T_2")
    print(f"  under O. The largest commutant in any case is M_2 (for 2 copies of")
    print(f"  one T-irrep), which hosts only su(2), NOT su(3).")
    print(f"  Joint commutant: dim = {dim_comm}.")
    print()
    print(f"  ✗ NEGATIVE: 2-dim multiplicity insufficient for su(3).")
elif iso == (6, 0, 0):
    print(f"  C_3-isotypic = (6, 0, 0) — V_Ram(Γ) is entirely C_3-trivial.")
    print(f"  Under O: 6·A_1 or 6·A_2 (or mix). Commutant = M_6.")
    print(f"  Joint commutant: dim = {dim_comm}.")
    print()
    print(f"  Algebraically room for su(3) in M_6 ⊃ M_3, but no canonical")
    print(f"  embedding picked out by {{B(Γ), U_{{C_3}}}} alone.")
    print(f"  STATUS: would require additional substrate operator to single out")
    print(f"          a 3-dim subspace.")
else:
    print(f"  C_3-isotypic = {iso} — mixed/non-uniform pattern.")
    print(f"  Joint commutant: dim = {dim_comm}.")
    print(f"  Decoding the O-decomposition requires the full character of O")
    print(f"  on V_Ram(Γ) (additional generators beyond U_{{C_3}}).")


print()
print("=" * 72)
print("OK: b4_color_vram_gamma_commutant_probe complete.")
print("=" * 72)
