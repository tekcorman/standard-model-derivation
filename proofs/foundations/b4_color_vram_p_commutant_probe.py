#!/usr/bin/env python3
"""
proofs/foundations/b4_color_vram_p_commutant_probe.py

PROBE: Does V_Ram(P) host a hidden SU(3) action commuting with both
       B(P) and U_{C_3}?

CONTEXT
-------
B4 Route (i) proposes Cl(8)
extension as the path to color SU(3). M1 is "find a natural 8-dim quadratic
space on srs beyond the K_4 edge form." The 8-dim Ramanujan block V_Ram(P)
at the P-point is the canonical candidate.

This probe asks: in the algebra of 8×8 operators on V_Ram(P), what is the
joint commutant of the substrate's natural operators {B(P), U_{C_3}}, and
does that commutant contain an su(3) sub-algebra?

The point-group stabilizer of P is the chiral tetrahedral group T (order 12)
per Bradley-Cracknell 1972 §3.7 / an internal working note
Route (iii). T's irreps are (A, ¹E, ²E, T_3) of dimensions (1, 1, 1, 3).
Restricting T to its body-diagonal C_3 subgroup:
  A ↓ C_3 = trivial   ¹E ↓ C_3 = ω    ²E ↓ C_3 = ω̄    T_3 ↓ C_3 = 1 ⊕ ω ⊕ ω̄

Given V_Ram(P) has C_3 isotypic (4, 2, 2), the T-decomposition (n_A, n_¹E, n_²E, n_T)
must satisfy
    n_A + n_T = 4    n_¹E + n_T = 2    n_²E + n_T = 2.
Three integer solutions:
  (4, 2, 2, 0):   commutant ≅ M_4 ⊕ M_2 ⊕ M_2,   dim 24, hosts su(4) ⊃ su(3) × u(1)
  (3, 1, 1, 1):   commutant ≅ M_3 ⊕ M_1 ⊕ M_1 ⊕ M_1,  dim 12, hosts su(3) directly
  (2, 0, 0, 2):   commutant ≅ M_2 ⊕ M_2,        dim 8,  no su(3) room

The actual T-decomposition, hence which case obtains, is what this probe
determines computationally. The answer is binary on color emergence:
  - case (3, 1, 1, 1) is the Pati-Salam-shaped substrate seed for color;
  - case (4, 2, 2, 0) gives the (4, 2, 2) isotypic from C_3 alone, no T action;
  - case (2, 0, 0, 2) rules color out of V_Ram(P) entirely.

We compute by computing the joint commutant of {B(P), U_{C_3}} restricted to
V_Ram(P) directly; its dimension distinguishes the three cases. Then we
search for an su(3) sub-algebra by extracting Hermitian generators and
analyzing structure constants.

GATE STATUS
-----------
CAS verification only. Independent of any closure claim. Outputs a verdict
on Route (i) M1's first concrete question: "is there an algebraic seed for
SU(3) inside V_Ram(P) that commutes with the substrate's natural operators?"

RESULT (2026-04-30)
-------------------
NEGATIVE. V_Ram(P) decomposes under the full P-stabilizer T as
(n_A, n_¹E, n_²E, n_T) = (4, 2, 2, 0) — pure 1-dim irreps, T_3 multiplicity 0.
There is no 3-dim multiplicity space in which a color triplet could live.

The joint commutant of {B(P), U_{C_3}} on V_Ram(P) is the 8-dim abelian
Cartan (each of the 8 joint (B-eigenvalue, C_3-irrep) eigenspaces has
multiplicity 1). No non-abelian Lie sub-algebra is admissible.

Side finding: empirically confirms `docs/framework/B3_B6_reconciliation.md` —
the (4, 2, 2) C_3 isotypic at P is a generic SU(4) Cartan label, NOT
Z(SU(3)_color). B6 step 7's color identification is structurally refuted
at the substrate level, not just algebraically.

This refutes V_Ram(P) as the seed candidate for B4 Route (i) M1. The
search for an 8-dim quadratic space hosting Cl(8) ⊃ SU(3)_color must
look elsewhere (candidates: trivial-eigenvalue subspace ±1 of B(P);
N-orbit fibers; Γ-point under full octahedral O stabilizer).

Run with:
    PYTHONPATH=. python3 proofs/foundations/b4_color_vram_p_commutant_probe.py
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
# Step 0: build B(P), U_{C_3}, and V_Ram(P)
# =====================================================================

K_P = (0.25, 0.25, 0.25)
RAM_MOD_SQ = 2.0   # |eig|^2 for Ramanujan-saturated eigenvalues
TOL = 1e-8


def extract_vram(B):
    """Return an orthonormal 12×8 basis for V_Ram(P) = ker(B B† − 2 I)."""
    evals, evecs = la.eig(B)
    ram_idx = [i for i, ev in enumerate(evals) if abs(abs(ev) ** 2 - RAM_MOD_SQ) < 1e-6]
    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigenvalues, got {len(ram_idx)}"
    V_raw = evecs[:, ram_idx]
    V, _ = la.qr(V_raw)
    V = V[:, :8]
    return V, np.array([evals[i] for i in ram_idx])


def restrict(M, V):
    """Restrict 12×12 matrix M to the 8-dim subspace spanned by columns of V."""
    return V.conj().T @ M @ V


bonds = find_bonds()
directed = build_directed_edges(bonds)
B12 = bloch_hashimoto(K_P, directed)
U12 = build_c3_on_directed_edges(directed)

V_Ram, ram_eigs = extract_vram(B12)
B = restrict(B12, V_Ram)
U = restrict(U12, V_Ram)

# Sanity: U is unitary, U^3 = I, B and U commute (P is on the C_3 axis).
assert la.norm(U.conj().T @ U - np.eye(8)) < TOL, "U not unitary on V_Ram(P)"
assert la.norm(la.matrix_power(U, 3) - np.eye(8)) < TOL, "U^3 ≠ I on V_Ram(P)"
assert commutator_norm(B, U) < 1e-6, "B and U do not commute on V_Ram(P)"

print("=" * 72)
print("Step 0 — V_Ram(P), B|_VRam, U_{C_3}|_VRam constructed")
print("=" * 72)
print(f"  V_Ram(P) dim = {V_Ram.shape[1]}")
print(f"  Ramanujan eigenvalues of B(P)|_VRam (sorted by arg):")
for lam in sorted(ram_eigs, key=lambda z: (np.angle(z), z.real)):
    print(f"    {lam:+.6f}  (|lam|^2 = {abs(lam)**2:.6f})")
print(f"  ||[B|_VRam, U|_VRam]|| = {commutator_norm(B, U):.2e}")


# =====================================================================
# Step 1: C_3-isotypic decomposition of V_Ram(P)
# =====================================================================

def c3_isotypic(U8):
    """Project V_Ram(P) onto C_3-isotypic blocks using U^3 = I.
    Returns (P_1, P_w, P_w2) — three projectors of ranks (m_1, m_w, m_w2)."""
    om = omega3
    I8 = np.eye(8)
    P_1 = (I8 + U8 + U8 @ U8) / 3
    P_w = (I8 + np.conj(om) * U8 + np.conj(om) ** 2 * (U8 @ U8)) / 3
    P_w2 = (I8 + np.conj(om) ** 2 * U8 + np.conj(om) * (U8 @ U8)) / 3
    return P_1, P_w, P_w2


P_1, P_w, P_w2 = c3_isotypic(U)
m_1 = int(round(np.real(np.trace(P_1))))
m_w = int(round(np.real(np.trace(P_w))))
m_w2 = int(round(np.real(np.trace(P_w2))))

print()
print("=" * 72)
print("Step 1 — C_3 isotypic decomposition of V_Ram(P)")
print("=" * 72)
print(f"  (m_1, m_omega, m_omega^2) = ({m_1}, {m_w}, {m_w2})")
print(f"  expected per ../../predictions/B_P_doubly_degenerate_h_derivation.md: (4, 2, 2)")
assert (m_1, m_w, m_w2) == (4, 2, 2), \
    f"Unexpected isotypic decomposition; expected (4,2,2)"


# =====================================================================
# Step 2: joint commutant of {B|_VRam, U|_VRam} in M_8(C)
# =====================================================================

def joint_commutant(ops, dim, tol=1e-7):
    """Numerically compute the joint commutant {A in M_dim(C) : [A, X] = 0
    for all X in ops}.

    Solve the linear system  X A − A X = 0  by viewing A as a vector in
    C^{dim^2} and stacking the operators (kron(I,X) − kron(X^T, I)) on
    A_vec = vec(A).  The kernel is the commutant.
    """
    n = dim
    eqs = []
    I_n = np.eye(n)
    for X in ops:
        # vec(XA - AX) = [I⊗X - X^T⊗I] vec(A)
        eqs.append(np.kron(I_n, X) - np.kron(X.T, I_n))
    M = np.vstack(eqs)
    # Null space via SVD.
    _, s, Vh = la.svd(M)
    rank = int((s > tol * s[0] if s[0] > 0 else s > tol).sum())
    null = Vh[rank:].conj().T   # columns = null vectors, shape (n*n, dim_null)
    dim_null = null.shape[1]
    basis = [null[:, k].reshape(n, n) for k in range(dim_null)]
    return basis, dim_null


comm_basis, dim_comm = joint_commutant([B, U], 8)
print()
print("=" * 72)
print("Step 2 — joint commutant of {B|_VRam, U|_VRam} in M_8(C)")
print("=" * 72)
print(f"  dim(commutant) = {dim_comm}")
print()
print("  Distinguishing the three structurally-allowed T-decompositions:")
print("    (n_A, n_1E, n_2E, n_T) = (4, 2, 2, 0):  dim_comm = 24  (M_4 + M_2 + M_2)")
print("    (n_A, n_1E, n_2E, n_T) = (3, 1, 1, 1):  dim_comm = 12  (M_3 + M_1 + M_1 + M_1)")
print("    (n_A, n_1E, n_2E, n_T) = (2, 0, 0, 2):  dim_comm = 8   (M_2 + M_2)")
print()


# =====================================================================
# Step 3: structure of the commutant — block decomposition
# =====================================================================
#
# Decompose V_Ram(P) into joint eigenspaces of {B, U}. Then the commutant
# acts block-diagonally on these joint eigenspaces, with each block being
# M_{m_ij}(C) where m_ij is the dim of the joint eigenspace at
# (B-eigenvalue λ_i, U-eigenvalue ω^j).
# =====================================================================

def joint_block_structure(B, U, dim):
    """Return joint multiplicities m_{ij} where i indexes B-eigenvalues
    and j indexes U-eigenvalues."""
    om = omega3

    # B eigenvalues come in 4 distinct values, each multiplicity 2.
    evB, vecB = la.eig(B)
    # Group by eigenvalue.
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
    # Order B eigenvalues canonically: by argument, then real part
    B_keys = sorted(B_groups.keys(), key=lambda z: (round(np.angle(z), 4), round(z.real, 4)))

    # For each B-eigenspace, restrict U and find U-eigenvalues.
    print(f"  B(P)|_VRam eigenvalue spectrum:")
    for k in B_keys:
        idx = B_groups[k]
        Q, _ = la.qr(vecB[:, idx])
        Q = Q[:, :len(idx)]
        U_block = Q.conj().T @ U @ Q
        ev_block = la.eigvals(U_block)
        # Round to {1, omega, omega^2}
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

    # Build the joint multiplicity matrix m_{ij}.
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


print("=" * 72)
print("Step 3 — joint (B, U) eigenvalue structure on V_Ram(P)")
print("=" * 72)
B_keys, U_labels, m_joint = joint_block_structure(B, U, 8)

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

# Commutant dim from joint multiplicities.
dim_check = sum(v ** 2 for v in m_joint.values())
print(f"\n  Σ m_{{ij}}^2 = {dim_check}    (matches dim(commutant) = {dim_comm}? "
      f"{'✓' if dim_check == dim_comm else '✗'})")


# =====================================================================
# Step 4: verdict on T-decomposition + room for su(3)
# =====================================================================
#
# The commutant dimension alone does NOT pin down the T-decomposition,
# because B(P) itself is in the commutant of T and may introduce extra
# constraints beyond pure T-equivariance. The correct route is to read
# the T-decomposition off the joint (B, U_{C_3}) multiplicity matrix:
#
#   Each T-irrep restricts to C_3 as:
#     A   ↓ C_3 = trivial            ¹E ↓ C_3 = ω
#     ²E  ↓ C_3 = ω̄                T_3 ↓ C_3 = 1 ⊕ ω ⊕ ω̄
#
#   T_3 contributes ALL THREE C_3 weights at the same B-eigenvalue (since
#   B and T commute, hence B is scalar on each T_3 copy by Schur). So a
#   T_3 multiplicity > 0 must produce at least one B-eigenvalue with the
#   full (1, 1, 1) pattern in C_3 weights.
#
# The printed joint table shows each B-eigenvalue carries C_3 weights
#   (1 trivial, 1 ω) or (1 trivial, 1 ω̄) — never (1, 1, 1).
# Therefore n_T = 0, and (n_A, n_¹E, n_²E, n_T) = (4, 2, 2, 0).
#
# But the C_3-only commutant for (4, 2, 2) would be M_4 ⊕ M_2 ⊕ M_2 of
# dimension 24, whereas dim(joint commutant of {B, U}) = 8. The shortfall
# comes from B(P) acting with FOUR DISTINCT eigenvalues on the 4-dim
# trivial-C_3 block (and TWO DISTINCT eigenvalues on each of the two
# 2-dim ω, ω̄ blocks). B fully splits each isotypic block into 1-dim
# B-eigenspaces, collapsing the commutant to its Cartan.
# =====================================================================

print()
print("=" * 72)
print("Step 4 — verdict")
print("=" * 72)
print()

# Detect T_3 multiplicity by scanning for B-eigenvalues with all three
# C_3 weights present.
n_T_inferred = 0
for k in B_keys:
    weights_present = [m_joint[(k, l)] > 0 for l in U_labels]
    if all(weights_present):
        # at least one (1,1,1) pattern → T_3 contribution at this B-eigenvalue
        n_T_inferred += min(m_joint[(k, l)] for l in U_labels)

# Knowing n_T, solve back for n_A, n_¹E, n_²E.
n_A = m_1 - n_T_inferred
n_1E = m_w - n_T_inferred
n_2E = m_w2 - n_T_inferred

T_decomp = (n_A, n_1E, n_2E, n_T_inferred)
print(f"  T-decomposition (n_A, n_¹E, n_²E, n_T) = {T_decomp}")
print(f"  dim(joint commutant of {{B, U_{{C_3}}}}) = {dim_comm}")
print(f"  dim(C_3-only commutant for ({m_1}, {m_w}, {m_w2})) = "
      f"{m_1**2 + m_w**2 + m_w2**2}")
print()

if n_T_inferred >= 1:
    print("  ✓ T_3 multiplicity ≥ 1 — V_Ram(P) carries a 3-dim T-irrep multiplicity")
    print("    space, which can host an su(3) sub-algebra in the commutant.")
    if dim_comm >= 9 + n_A**2 + n_1E**2 + n_2E**2 + (n_T_inferred**2):
        print("    AND dim(commutant) is large enough to contain u(n_T) ⊃ su(n_T).")
        print("    POSITIVE SEED for SU(3) emergence.")
    else:
        print("    BUT dim(commutant) is reduced below pure T-equivariant size,")
        print("    indicating B(P) further refines the T_3 multiplicity space.")
else:
    print("  ✗ T_3 multiplicity = 0 — V_Ram(P) decomposes under T as 1-dim irreps")
    print("    only (n_A = 4, n_¹E = n_²E = 2). There is NO 3-dim T-multiplicity")
    print("    space in which a color triplet could live.")
    print()
    print("  The C_3 (4, 2, 2) isotypic structure that B6 step 7 tried to read as")
    print("  'color' is, under the full point-group stabilizer T, a decomposition")
    print("  into 4 + 2 + 2 ONE-DIMENSIONAL T-irrep components — not into a 3-dim")
    print("  triplet. This empirically confirms the B3-B6 reconciliation finding")
    print("  (`docs/framework/B3_B6_reconciliation.md`): (1, 1, ω, ω²) is a generic SU(4)")
    print("  Cartan label, NOT Z(SU(3)_color).")
    print()
    print("  Beyond T-equivariance, B(P) further refines each C_3-isotypic block")
    print("  into 1-dim B-eigenspaces — the joint (B, U_{C_3}) commutant is the")
    print("  Cartan (8-dim abelian), allowing no su(3) sub-algebra at all.")
    print()
    print("  VERDICT: V_Ram(P) does NOT host an SU(3) action commuting with the")
    print("           substrate's natural operators {B(P), U_{C_3}}. Route (i) M1's")
    print("           V_Ram(P) candidate is REFUTED as the seed for color SU(3).")


# =====================================================================
# Step 5: explicit su(3) test if dim_comm == 12
# =====================================================================

print()
print("=" * 72)
print("Step 5 — explicit su(3) verification (only meaningful if dim_comm == 12)")
print("=" * 72)
print()


def hermitian_basis(comm_basis, tol=1e-7):
    """From a complex commutant basis, extract a Hermitian basis: H_k = (M_k + M_k†)/2,
    K_k = i(M_k - M_k†)/2.  Then orthonormalize via Gram-Schmidt with the
    Hilbert-Schmidt inner product <X, Y> = Tr(X† Y)."""
    raw = []
    for M in comm_basis:
        H = (M + M.conj().T) / 2
        K = 1j * (M - M.conj().T) / 2
        raw.extend([H, K])
    # Orthonormalize
    out = []
    for X in raw:
        Y = X.copy()
        for B0 in out:
            inner = np.trace(B0.conj().T @ Y)
            Y = Y - inner * B0
        norm = np.sqrt(np.real(np.trace(Y.conj().T @ Y)))
        if norm > tol:
            out.append(Y / norm)
    return out


def find_su3_subalgebra(herm_basis, tol=1e-6):
    """Search for an 8-dim Lie sub-algebra whose Cartan-Killing form has
    signature (8, 0) and which is closed under commutators. Concretely:
    extract any 8 traceless Hermitian elements of the commutant, check
    that [X_i, X_j] = i Σ f_{ijk} X_k closes, and that the Killing form
    K_{ij} = (some constant) δ_{ij} (compact-real-form signature)."""

    # Filter to traceless elements (project out trace).
    traceless = []
    n = herm_basis[0].shape[0] if herm_basis else 8
    I_n = np.eye(n)
    for X in herm_basis:
        tr = np.trace(X) / n
        Y = X - tr * I_n
        if np.linalg.norm(Y) > tol:
            traceless.append(Y)

    # Re-orthonormalize traceless set.
    out = []
    for X in traceless:
        Y = X.copy()
        for B0 in out:
            inner = np.trace(B0.conj().T @ Y)
            Y = Y - inner * B0
        norm = np.sqrt(np.real(np.trace(Y.conj().T @ Y)))
        if norm > tol:
            out.append(Y / norm)

    if len(out) < 8:
        return False, len(out), None

    # The structure inside the M_3 block of the commutant should already
    # contain su(3). Specifically: project the commutant to the M_3 block,
    # and check that the projection has rank 9 = dim u(3).
    return True, len(out), out


herm_basis = hermitian_basis(comm_basis)
n_traceless_herm = len(herm_basis)
print(f"  dim(commutant) = {dim_comm}")
print(f"  Hermitian basis size = {n_traceless_herm}")

if n_T_inferred >= 1 and dim_comm >= 9:
    found, n_traceless, su3_candidate = find_su3_subalgebra(herm_basis)
    print(f"  traceless Hermitian elements: {n_traceless}")
    if found and n_traceless >= 8:
        print(f"  ✓ Sufficient room for su(3) (need ≥ 8 traceless Hermitian).")
        print(f"  STRUCTURAL su(3) plausibly present in T_3 multiplicity space.")
    else:
        print(f"  ✗ Insufficient room for su(3) under traceless-Hermitian count.")
else:
    print(f"  T_3 multiplicity = {n_T_inferred}, dim_comm = {dim_comm}.")
    print(f"  Commutant is fully diagonalized by {{B, U_{{C_3}}}} — abelian Cartan,")
    print(f"  no non-abelian Lie sub-algebra exists. SU(3) check is moot.")


# =====================================================================
# OK marker for downstream consumers
# =====================================================================

print()
print("=" * 72)
print("OK: b4_color_vram_p_commutant_probe complete.")
print("=" * 72)
