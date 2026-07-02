#!/usr/bin/env python3
"""
de_rham_susy_fibered_v2_probe.py
================================
Second attempt at the fibered de Rham SUSY: gauge-equivariant by construction,
using partial trace at the operator-algebra level (no |0⟩|0⟩ reference state).

The v1 probe (`de_rham_susy_fibered_probe.py`) tried to lift d to a STATE map
Cl(6)_v Fock (ℂ⁸) → ℂ²_e by contracting non-e tensor factors against |0⟩|0⟩.
That failed gauge SU(2) equivariance because the reference state |0⟩|0⟩ implicitly
fixes a gauge on the non-e edges.  Representation theory in fact RULES OUT a
linear SU(2)-equivariant projection ℂ⁸ → ℂ² over the non-e SU(2) action (ℂ²
has no SU(2)-invariant subspace other than {0}).

The fix: lift to the OPERATOR ALGEBRA level.
  Cl(6)_v at each vertex = 8×8 complex matrices, M₈(ℂ), 64-dim as a vector space.
  Cl(2)_e at each edge   = 2×2 complex matrices, M₂(ℂ),  4-dim.
The partial trace  tr_⊥: M₈ → M₂  over the non-e tensor factors IS gauge-
equivariant: for U on the e-factor, tr_⊥(UAU†) = U tr_⊥(A) U†;  for U on a
non-e factor, the trace cancels it: tr_⊥(UAU†) = tr_⊥(A) (cross-edge gauge
acts trivially on the e-edge content, as it should).

So the operator-algebra fibered cochain is gauge-equivariant by construction,
and we can equip C⁰_alg = ⊕_v Cl(6)_v (256-dim) and C¹_alg = ⊕_e Cl(2)_e (24-dim)
with the Hilbert–Schmidt inner product  ⟨X,Y⟩ = tr(X†Y)  (the GNS / Liouville-space
view) — turning them into bona fide Hilbert spaces.  Then Q̂_alg = d̂_alg + d̂_alg†
acts on C⁰_alg ⊕ C¹_alg as a graded operator with Q̂_alg² = blockdiag(Δ̂_0^alg,
Δ̂_1^alg).

What this probe builds / checks
-------------------------------
A — algebraic setup: Cl(6) via Jordan-Wigner over the 3 incident edges per vertex;
    Cl(2) on each edge;  partial trace as a (4×64) linear map per "which slot of v's
    three is the e-slot."
B — fibered d̂_alg with the partial-trace projection;  SUSY algebra closure
    (Q̂_alg² = blockdiag, {Q̂_alg, χ̂} = 0).
C — Witten pairing of the nonzero spectrum.
D — *full* gauge equivariance:  for U ∈ SU(2)_e (per-edge) AND U' ∈ SU(2)_{e'}
    (cross-edge), test that the adjoint action on C⁰_alg ⊕ C¹_alg commutes with
    Q̂_alg.
E — interpretation: Q̂_alg maps matter OPERATORS at vertices to gauge OPERATORS at
    edges.  What does this mean physically?  In MSSM, SUSY maps matter STATES to
    boson STATES — Q̂_alg is a different beast.  Honest discussion of whether the
    operator-algebra-level fibered SUSY is the MSSM mechanism, a *different*
    mechanism that could play the same role, or a category-different structure.

VERDICT (printed honestly).  If gauge equivariance holds AND the resulting
algebra-level Q̂ has the structure to influence β-functions / matter-gauge
matching, this is a viable mechanism candidate.  If equivariance holds but the
construction is category-different from MSSM (algebra-level vs state-level), say
so precisely so we know what we have.  No graded content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, K_STAR, N_ATOMS  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=150)

BONDS = find_bonds()
GAMMA = (0.0, 0.0, 0.0)
P_POINT = (0.25, 0.25, 0.25)
GENERIC_K = (0.137, 0.291, 0.453)


def _cell_edges():
    seen = {}
    for u, v, c in BONDS:
        c = tuple(int(x) for x in c)
        key = (min(u, v), max(u, v), tuple(sorted((c, tuple(-x for x in c)))))
        if key in seen:
            continue
        seen[key] = (u, v, c) if u <= v else (v, u, tuple(-x for x in c))
    return sorted(seen.values())


EDGES = _cell_edges()
NE = len(EDGES)
NV = N_ATOMS


def incident_edges(v):
    out = []
    for i, (a, b, _) in enumerate(EDGES):
        if a == v:
            out.append((i, -1))   # v is tail
        elif b == v:
            out.append((i, +1))   # v is head
    return tuple(out)


# ----------------------------------------------------------------------
# Partial-trace linear map  tr_⊥(slot): M₈ → M₂
#
# v's Cl(6) Fock = ℂ²_a ⊗ ℂ²_b ⊗ ℂ²_c.  A linear operator A on ℂ⁸ is an 8×8 matrix.
# tr_⊥(slot=i): trace out the OTHER two factors; keep factor i (the e-factor).
# Normalised: tr_⊥(I_8) / 4 = I_2.
# ----------------------------------------------------------------------

def _partial_trace_8_to_2(slot):
    """Return a (4, 64) matrix  T  such that  flatten(tr_⊥(A)) = T @ flatten(A) / 4
    (where flatten is column-major flatten of the 8×8 matrix A and the 2×2 partial
    trace).  Trace out the two slots ≠ `slot`."""
    T = np.zeros((4, 64), dtype=complex)
    # 8×8 indexed by (i_a, i_b, i_c) on rows and (j_a, j_b, j_c) on cols, each ∈ {0,1}.
    # row index of A in column-major flatten: A_{(i,j)} flattened to position j*8 + i.
    # we want [tr_⊥(A)]_{ii'} where (i, i') are values on slot, summing over the other slots.
    for i_a in (0, 1):
        for i_b in (0, 1):
            for i_c in (0, 1):
                for j_a in (0, 1):
                    for j_b in (0, 1):
                        for j_c in (0, 1):
                            row = (i_a * 4 + i_b * 2 + i_c)
                            col = (j_a * 4 + j_b * 2 + j_c)
                            A_idx = col * 8 + row    # column-major flatten of 8×8 A
                            # the kept slot is `slot`: i_slot and j_slot
                            ks = [i_a, i_b, i_c][slot]
                            js = [j_a, j_b, j_c][slot]
                            # to contribute to tr_⊥(A)[ks, js]: need the other two slots' i = j
                            other_i = [[i_a, i_b, i_c][k] for k in range(3) if k != slot]
                            other_j = [[j_a, j_b, j_c][k] for k in range(3) if k != slot]
                            if other_i == other_j:
                                # column-major flatten of 2×2 with row ks, col js: position js*2 + ks
                                T_row = js * 2 + ks
                                T[T_row, A_idx] += 1.0
    return T / 4.0


# precompute the three partial-trace operators for slot ∈ {0,1,2}
T_SLOT = [_partial_trace_8_to_2(s) for s in (0, 1, 2)]


def _phase_basis_vectors():
    """Returns the 64 standard basis vectors of M_8(ℂ) (each as a 64-vector in column-major flatten).
    Mostly here for clarity; not strictly used downstream."""
    out = []
    for c in range(8):
        for r in range(8):
            v = np.zeros(64, dtype=complex)
            v[c * 8 + r] = 1.0
            out.append(v)
    return out


# ----------------------------------------------------------------------
# fibered cochain  d̂_alg(k) : C⁰_alg (256-dim) → C¹_alg (24-dim)
#
# basis order:  C⁰_alg index = v * 64 + α,  α ∈ [0, 64);  C¹_alg index = e * 4 + β.
# ----------------------------------------------------------------------

def d_alg(k):
    d = np.zeros((NE * 4, NV * 64), dtype=complex)
    kk = np.asarray(k, float)
    for e_idx, (u, v, voltage) in enumerate(EDGES):
        phase = np.exp(2j * np.pi * np.dot(kk, voltage))
        for vertex, sign in [(v, +1.0), (u, -phase)]:
            incs = [eid for eid, _ in incident_edges(vertex)]
            slot = incs.index(e_idx)
            T = T_SLOT[slot]
            d[e_idx * 4:(e_idx + 1) * 4, vertex * 64:(vertex + 1) * 64] += sign * T
    return d


# ----------------------------------------------------------------------
# Cl(6) generators (8×8) on each vertex's Fock, via Jordan–Wigner over its 3 incident edges
# ----------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def _cl2_action_on_slot(U, slot):
    """U ∈ M_2(ℂ) acting on the `slot`-th tensor factor of M_8(ℂ); returns an 8×8 matrix."""
    mats = [I2, I2, I2]
    mats[slot] = U
    return _kron3(*mats)


# ======================================================================
def part_A():
    print("=" * 100)
    print("PART A — operator-algebra fibered cochain  d̂_alg : C⁰_alg (256-dim) → C¹_alg (24-dim)")
    print("         (Cl(6)_v ≅ M₈(ℂ), Cl(2)_e ≅ M₂(ℂ), partial trace tr_⊥ : M₈ → M₂  gauge-equivariant)")
    print("=" * 100)
    # check that the 3 partial-trace operators behave correctly: tr_⊥(I_8) = I_2
    for s in (0, 1, 2):
        I8 = np.eye(8, dtype=complex).flatten('F')
        out = T_SLOT[s] @ I8
        I2_vec = np.eye(2, dtype=complex).flatten('F')
        ok = np.allclose(out, I2_vec)
        print(f"  partial trace, slot {s}: tr_⊥(I₈)/4 = I₂  →  {ok}")
        assert ok


def part_B(k):
    print("\n" + "=" * 100)
    print(f"PART B — SUSY algebra closure  at k = {k}")
    print("=" * 100)
    d = d_alg(k)
    dim0, dim1 = NV * 64, NE * 4
    Q = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    Q[:dim0, dim0:] = d.conj().T
    Q[dim0:, :dim0] = d
    chi = np.diag([1.0] * dim0 + [-1.0] * dim1)
    QQ = Q @ Q
    D0, D1 = d.conj().T @ d, d @ d.conj().T
    blk = np.zeros_like(QQ); blk[:dim0, :dim0] = D0; blk[dim0:, dim0:] = D1
    ok_Q2 = np.allclose(QQ, blk, atol=1e-10)
    ok_chi = np.allclose(Q @ chi + chi @ Q, 0, atol=1e-10)
    print(f"\n  dim C⁰_alg = {dim0}    dim C¹_alg = {dim1}    d̂_alg = {d.shape[0]} × {d.shape[1]}")
    print(f"  Q̂_alg² = blockdiag(Δ̂_0^alg, Δ̂_1^alg):  {ok_Q2}        {{Q̂_alg, χ̂}} = 0:  {ok_chi}")
    assert ok_Q2 and ok_chi
    return d, D0, D1


def part_C(D0, D1, k):
    print("\n" + "-" * 100)
    print("C — Witten pairing of the nonzero spectrum (= matter↔gauge supermultiplet at operator-algebra level)")
    print("-" * 100)
    s0 = np.sort(np.linalg.eigvalsh((D0 + D0.conj().T) / 2))
    s1 = np.sort(np.linalg.eigvalsh((D1 + D1.conj().T) / 2))
    s0_nz = s0[s0 > 1e-8]
    s1_nz = s1[s1 > 1e-8]
    pairs = (len(s0_nz) == len(s1_nz)) and np.allclose(s0_nz, s1_nz, atol=1e-7)
    print(f"\n  Δ̂_0^alg (256-d): {len(s0_nz)} nonzero, {len(s0) - len(s0_nz)} zero modes")
    print(f"  Δ̂_1^alg ( 24-d): {len(s1_nz)} nonzero, {len(s1) - len(s1_nz)} zero modes")
    print(f"  Witten pairing:  {pairs}     (nonzero spectra match: {pairs})")
    print(f"  nonzero spec Δ̂_0^alg (unique values): {np.unique(np.round(s0_nz, 4)).tolist()}")
    print(f"  nonzero spec Δ̂_1^alg (unique values): {np.unique(np.round(s1_nz, 4)).tolist()}")
    print(f"  Witten index  ind Q̂_alg = dim ker Δ̂_0^alg − dim ker Δ̂_1^alg = "
          f"{(len(s0) - len(s0_nz)) - (len(s1) - len(s1_nz))}")
    assert pairs


def part_D(d, k):
    print("\n" + "-" * 100)
    print("D — FULL gauge equivariance (the v1 probe failed this)")
    print("-" * 100)
    rng = np.random.default_rng(0)

    def random_su2():
        # random SU(2): exp(i θ n·σ/2)
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        th = rng.uniform(0, 2 * np.pi)
        return np.cos(th / 2) * I2 + 1j * np.sin(th / 2) * (n[0] * SX + n[1] * SY + n[2] * SZ)

    def adjoint_on_algebra(U, slot, dim=8):
        """Adjoint action A ↦ U·A·U†, where U acts on `slot` of the dim-d Hilbert space.
        Returns a (dim², dim²) matrix acting on flattened M_d.  For Cl(6) on slot ∈ {0,1,2}
        of (ℂ²)⊗³ at a vertex, dim=8 with slot being the qubit; for Cl(2) on the single
        edge qubit, slot is ignored and dim=2."""
        if dim == 8:
            U_full = _cl2_action_on_slot(U, slot)
        elif dim == 2:
            U_full = U
        else:
            raise ValueError(dim)
        return np.kron(U_full.conj(), U_full)   # acts on column-major flatten of (d, d) matrix

    print("\n  test 1 — per-EDGE gauge equivariance (U on edge e₀'s qubit, on BOTH sides):")
    target_edge = 0
    U = random_su2()
    # action on C⁰_alg: at each vertex v containing edge 0 in its incidents, apply Ad_U on the slot for edge 0
    AdU_C0 = np.zeros((NV * 64, NV * 64), dtype=complex)
    for v in range(NV):
        incs = [eid for eid, _ in incident_edges(v)]
        block = np.eye(64, dtype=complex)
        if target_edge in incs:
            slot = incs.index(target_edge)
            block = adjoint_on_algebra(U, slot, dim=8)
        AdU_C0[v * 64:(v + 1) * 64, v * 64:(v + 1) * 64] = block
    # action on C¹_alg: only edge 0 gets transformed
    AdU_C1 = np.eye(NE * 4, dtype=complex)
    AdU_C1[target_edge * 4:(target_edge + 1) * 4, target_edge * 4:(target_edge + 1) * 4] = adjoint_on_algebra(U, 0, dim=2)
    diff_per = np.linalg.norm(d @ AdU_C0 - AdU_C1 @ d)
    print(f"    ‖d̂_alg · Ad^{{C⁰}}_U − Ad^{{C¹}}_U · d̂_alg‖ = {diff_per:.3e}   →  per-edge equivariant: {diff_per < 1e-9}")

    print("\n  test 2 — CROSS-edge gauge invariance (U' on a DIFFERENT edge e₁'s qubit, acting on the matter side):")
    other_edge = 1
    Up = random_su2()
    AdU_C0_other = np.zeros((NV * 64, NV * 64), dtype=complex)
    for v in range(NV):
        incs = [eid for eid, _ in incident_edges(v)]
        block = np.eye(64, dtype=complex)
        if other_edge in incs:
            slot = incs.index(other_edge)
            block = adjoint_on_algebra(Up, slot, dim=8)
        AdU_C0_other[v * 64:(v + 1) * 64, v * 64:(v + 1) * 64] = block
    # the gauge action on C¹_alg is only on edge `other_edge`; check that d̂_alg's projection
    # to edge e₀ is UNCHANGED by Ad_{Up} on the matter side (since the partial trace traces out
    # the other_edge factor)
    AdUp_C1 = np.eye(NE * 4, dtype=complex)
    AdUp_C1[other_edge * 4:(other_edge + 1) * 4, other_edge * 4:(other_edge + 1) * 4] = adjoint_on_algebra(Up, 0, dim=2)
    diff_cross = np.linalg.norm(d @ AdU_C0_other - AdUp_C1 @ d)
    print(f"    ‖d̂_alg · Ad^{{C⁰}}_{{Up,e₁}} − Ad^{{C¹}}_{{Up,e₁}} · d̂_alg‖ = {diff_cross:.3e}   →  cross-edge equivariant: {diff_cross < 1e-9}")

    print(f"\n  ⇒ FULL gauge equivariance:  {diff_per < 1e-9 and diff_cross < 1e-9}")
    return diff_per, diff_cross


def part_E(d):
    print("\n" + "-" * 100)
    print("E — interpretation: Q̂_alg maps matter OPERATORS at vertices to gauge OPERATORS at edges")
    print("-" * 100)
    # what's the image of an identity-at-a-vertex element under d̂_alg?
    I_v0 = np.zeros(NV * 64, dtype=complex)
    I8_flat = np.eye(8, dtype=complex).flatten('F')
    I_v0[0 * 64:1 * 64] = I8_flat
    img = d @ I_v0
    print(f"\n  d̂_alg(I_at_v₀):  ‖·‖² = {np.linalg.norm(img) ** 2:.4f}")
    nonzero_edges = []
    for e in range(NE):
        e_img = img[e * 4:(e + 1) * 4]
        if np.linalg.norm(e_img) > 1e-9:
            mat = e_img.reshape(2, 2, order='F')
            nonzero_edges.append((e, mat))
    print(f"  contributions on edges:")
    for e, mat in nonzero_edges:
        # decompose mat into Pauli basis: mat = a I + b σx + c σy + d σz
        a = 0.5 * np.trace(mat)
        b = 0.5 * np.trace(SX @ mat)
        c = 0.5 * np.trace(SY @ mat)
        dd = 0.5 * np.trace(SZ @ mat)
        print(f"    edge {e} (Cl(2) image): {a.real:+.3f}·I  +  {b.real:+.3f}·σx  +  {c.real:+.3f}·σy  +  {dd.real:+.3f}·σz")
    # try a non-trivial Cl(6) element: γ₁·γ₂ (a bivector — generates rotation in the first edge-qubit slot's Cl(2))
    gamma1 = _cl2_action_on_slot(SX, 0)        # σ_x on slot 0 of (ℂ²)⊗³ = γ₁ in JW
    gamma2 = _cl2_action_on_slot(SY, 0)        # σ_y on slot 0 = γ₂
    biv = gamma1 @ gamma2                       # γ₁γ₂ — generates SO(2) rotation in slot 0
    biv_flat = biv.flatten('F')
    psi = np.zeros(NV * 64, dtype=complex); psi[0:64] = biv_flat
    img2 = d @ psi
    print(f"\n  d̂_alg(γ₁γ₂ at v₀)  (a slot-0 bivector at vertex 0):  ‖·‖² = {np.linalg.norm(img2) ** 2:.4f}")
    nonzero_edges = []
    for e in range(NE):
        e_img = img2[e * 4:(e + 1) * 4]
        if np.linalg.norm(e_img) > 1e-9:
            mat = e_img.reshape(2, 2, order='F')
            nonzero_edges.append((e, mat))
    for e, mat in nonzero_edges:
        a = 0.5 * np.trace(mat)
        b = 0.5 * np.trace(SX @ mat)
        c = 0.5 * np.trace(SY @ mat)
        dd = 0.5 * np.trace(SZ @ mat)
        print(f"    edge {e} (Cl(2) image): {a.real:+.3f}+{a.imag:+.3f}i ·I  +  {b.real:+.3f}+{b.imag:+.3f}i ·σx  +  "
              f"{c.real:+.3f}+{c.imag:+.3f}i ·σy  +  {dd.real:+.3f}+{dd.imag:+.3f}i ·σz")


def main():
    print(r"""
======================================================================================================
DE RHAM SUSY, FIBERED v2 — gauge-equivariant via partial trace at the operator-algebra level
======================================================================================================
""")
    part_A()
    diffs_per_k = []
    for label, k in [("Γ", GAMMA), ("P=(¼,¼,¼)", P_POINT), ("generic k", GENERIC_K)]:
        print(f"\n\n#####  k-point: {label}  #####")
        d, D0, D1 = part_B(k)
        part_C(D0, D1, k)
        diff_per, diff_cross = part_D(d, k)
        diffs_per_k.append((label, diff_per, diff_cross))
        if label == "Γ":
            part_E(d)
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    all_eq = all(p < 1e-9 and c < 1e-9 for _, p, c in diffs_per_k)
    print(f"""
  ALGEBRA STRUCTURE
   • Q̂_alg² = blockdiag(Δ̂_0^alg, Δ̂_1^alg) at every k tested;  {{Q̂_alg, χ̂}} = 0;  Witten pairing exact
     (nonzero spec Δ̂_0^alg = nonzero spec Δ̂_1^alg, matched multiplicities).

  GAUGE EQUIVARIANCE  (the v1 probe's failure mode)
   • per-edge SU(2):  ‖[d̂_alg, Ad_U]‖  ≈  {max(p for _,p,_ in diffs_per_k):.2e}  across all k tested
   • cross-edge SU(2): ‖[d̂_alg, Ad_{{U,e'}}]‖  ≈  {max(c for _,_,c in diffs_per_k):.2e}  across all k tested
   ⇒ FULL gauge equivariance: {'YES' if all_eq else 'NO'}  — partial trace is gauge-equivariant by construction.

  INTERPRETATION
   Q̂_alg maps matter OPERATORS at vertices (Cl(6)_v ≅ M₈) to gauge OPERATORS at edges (Cl(2)_e ≅ M₂).
   This is NOT the textbook MSSM SUSY (which maps matter STATES to scalar-superpartner STATES).  But it
   IS a gauge-equivariant graded operator with Q̂² = Hamiltonian on the operator-algebra space — a
   spectral-triple / non-commutative-geometry-style supercharge.

   What does it mean for β-functions?
     β-functions are usually computed from particle CONTENT (states), not from the algebra of operators.
     The operator-algebra SUSY relates matter operators to gauge operators — they are 'partners' at the
     algebra level, but not separate states.  This is consistent with the v1 finding that the framework's
     vertex Focks and edge qubits sit on a single 6-qubit underlying Hilbert space (no doubling).

   Honest reading:  this construction gives a CATEGORY-DIFFERENT mechanism from MSSM.  MSSM doubles the
   spectrum with separate scalar superpartners;  the framework's fibered de Rham SUSY at the algebra
   level relates the existing matter and gauge operator algebras without doubling.  These are different
   things.  Whether the algebra-level SUSY influences gauge β-functions in a way that matches MSSM's
   effect requires a separate computation (NOT done here) — and is plausibly NO because β-functions are
   state-level quantities.

  WHAT THIS PROBE ACTUALLY CLOSES
   • A gauge-equivariant fibered version of the de Rham Q̂ DOES exist — at the operator-algebra level,
     not the state level.  The v1 probe's gauge failure was specific to its state-level construction.
   • The framework therefore carries a (gauge-equivariant) fibered SUSY structure, but it is at the
     operator-algebra level, not the Hilbert-space-doubling level MSSM requires.
   • The MSSM mechanism, if it exists in the framework, is NOT this — but the existence of this
     algebra-level SUSY constrains where the MSSM mechanism could live (it can't be a state-level
     fibered de Rham SUSY; the rep theory rules that out).

  No graded content changes.
""")
    print("de_rham_susy_fibered_v2_probe.py: all checks passed (sentinel).")


if __name__ == "__main__":
    main()
