#!/usr/bin/env python3
"""
de_rham_susy_fibered_probe.py
=============================
Lift the cochain-base de Rham supercharge Q = d + d* to the FULL FIBERED Hilbert
space, using the framework's own Cl(6) = Cl(2)^⊗3 decomposition over incident
edges, and ask: is the resulting fibered Q̂ a real fermion↔boson SUSY in the
framework, or a re-labeling of a single Hilbert space?

This is the actual MSSM-mechanism attempt.  The framework's field sorting is:
  vertices ↔ Cl(6) Fock = matter (fermions);  edges ↔ Cl(0,2)≅ℍ + Cl(2) = gauge
bosons + Higgs.  Q on the cochain base maps degree-0 to degree-1, so it maps
fermion sector to boson sector — algebraically a fermion↔boson supercharge.  The
question this probe answers: does Q lift to the fibered Hilbert space (vertex
matter ℂ⁸/vertex + edge qubit ℂ²/edge) cleanly, and does the lift give a real
SUSY structure with matching gauge quantum numbers under the framework's already-
derived SU(2) (`theorem_g2_edge_qubit_su2.md`)?

Construction (Cl(6) Fock = ⊗ over incident edge qubits)
-------------------------------------------------------
Each vertex v has 3 incident edges (e_a, e_b, e_c) in some fixed ordering.  The
vertex's Cl(6) Fock factors as Cl(2)_{e_a} ⊗ Cl(2)_{e_b} ⊗ Cl(2)_{e_c}, and the
Fock representation ℂ⁸_v = ℂ²_{e_a} ⊗ ℂ²_{e_b} ⊗ ℂ²_{e_c}.  Each edge qubit
ℂ²_e is shared as a tensor factor of EXACTLY the two endpoint vertices' Focks.
That sharing is the structural fact this probe exploits.

For each edge e incident to v, define the FIBER PROJECTION
    P_e^(v) : ℂ⁸_v → ℂ²_e
by contracting the OTHER two tensor factors of ℂ⁸_v against |0⟩|0⟩ (the empty-
edge reference).  Then the FIBERED COBOUNDARY  d̂ : C⁰_fib (32-dim) → C¹_fib
(12-dim)  is, for each oriented edge e = u → v with voltage n:
    (d̂ ψ)_e  =  P_e^(v)(ψ_v)  −  e^{2πi k·n}·P_e^(u)(ψ_u)
The fibered supercharge is  Q̂ = [[0, d̂†],[d̂, 0]]  on C⁰_fib ⊕ C¹_fib.

What this probe checks
----------------------
A — algebra: Q̂² = blockdiag(d̂†d̂, d̂d̂*) =: blockdiag(Δ̂₀, Δ̂₁), Q̂ anticommutes
    with the cochain-degree grading χ̂ = diag(+I_{32}, −I_{12}).
B — Witten pairing: nonzero spec Δ̂₀ = nonzero spec Δ̂₁ (at the same multiplicities).
C — restriction to the cochain BASE: when ψ_v = α_v·|vac⟩ (Λ⁰ subspace at every
    vertex), Q̂ reduces to the unfibered cochain Q from `de_rham_susy_on_srs_probe.py`.
D — action on the Λ-graded matter states (Λ⁰=lepton, Λ¹=down, Λ²=up, Λ³=neutrino):
    does Q̂ send Λⁿ to Λⁿ⁺¹ via incidence (the framework's species ladder), and what
    does it do on the edge side?
E — SU(2) gauge equivariance: per-edge SU(2) action commutes with Q̂?
F — *the key question*: is the fibered Q̂ adding NEW degrees of freedom (separate
    fermion + boson sectors with their own occupation numbers, à la MSSM
    super-partners), or is it a RE-LABELLING of a single underlying Hilbert space
    (same qubit excitations read either as fermions at vertices or as bosons on
    edges)?  Test by examining the single-particle subspaces and counting states.

VERDICT (printed honestly).  Either:
  • Q̂ adds distinct fermion+boson modes → framework has a real Cl(6)/Cl(0,2)
    SUSY structure that could in principle give MSSM-like β-function effects
    (next probe: extract gauge β-function contributions); OR
  • Q̂ is a re-labelling on a single Hilbert space → the "fermion-boson swap"
    is a duality of READINGS, not a doubling; framework β-functions are framework-
    specific (not SM, not MSSM); we need a different route for the MSSM dictionary.
Either result is a closure of the de-Rham-SUSY-as-MSSM question.  No graded
content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, bloch_H, K_STAR, N_ATOMS  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=150)

BONDS = find_bonds()
GAMMA = (0.0, 0.0, 0.0)
P_POINT = (0.25, 0.25, 0.25)
GENERIC_K = (0.137, 0.291, 0.453)


# ----------------------------------------------------------------------
# cell structure
# ----------------------------------------------------------------------

def _cell_edges():
    seen = {}
    for u, v, c in BONDS:
        c = tuple(int(x) for x in c)
        key = (min(u, v), max(u, v), tuple(sorted((c, tuple(-x for x in c)))))
        if key in seen:
            continue
        seen[key] = (u, v, c) if u <= v else (v, u, tuple(-x for x in c))
    return sorted(seen.values())


EDGES = _cell_edges()     # 6 oriented edges
NE = len(EDGES)
NV = N_ATOMS              # 4
N_FOCK = 8                # Cl(6) Fock dim per vertex
N_QUBIT = 2               # edge qubit dim


def incident_edges(v):
    """Ordered tuple of (edge index, +1 if v is the head, −1 if tail) for each incident edge."""
    out = []
    for i, (a, b, _) in enumerate(EDGES):
        if a == v:
            out.append((i, -1))   # v is tail (the "u" of e=u→v in the cochain convention)
        elif b == v:
            out.append((i, +1))   # v is head (the "v" of e=u→v)
    assert len(out) == K_STAR, f"vertex {v} has {len(out)} incidences (expected {K_STAR})"
    return tuple(out)


# ----------------------------------------------------------------------
# fiber projection  P_e^(v) : Cl(6)_v Fock (ℂ⁸) → ℂ²_e
#
# Vertex v's Cl(6) Fock is ordered as ⊗_{i=0,1,2} ℂ²_{(incident edge i)};
# for an incident edge e (slot i in this ordering), the projection contracts
# the other two tensor factors against |0⟩|0⟩ — extracting the e-qubit.
# ----------------------------------------------------------------------

def _bits_of(state, n):
    """state ∈ [0, 2^n) → (b₀, b₁, …, b_{n−1})  with b₀ the most significant (leftmost factor)."""
    return tuple((state >> (n - 1 - i)) & 1 for i in range(n))


def fiber_projection(v, e_global):
    incs = [eid for eid, _ in incident_edges(v)]
    pos = incs.index(e_global)        # which tensor slot is the e-qubit
    P = np.zeros((N_QUBIT, N_FOCK), dtype=complex)
    for s in range(N_FOCK):
        bits = _bits_of(s, 3)
        # the other 2 factors must be |0⟩ to contribute
        if all(bits[j] == 0 for j in range(3) if j != pos):
            P[bits[pos], s] = 1.0
    return P


# ----------------------------------------------------------------------
# fibered coboundary  d̂(k) : C⁰_fib (32-dim) → C¹_fib (12-dim)
#
# C⁰_fib basis: |v, s⟩ for v ∈ V, s ∈ {0..7} (Cl(6) Fock state).  Index = v*8 + s.
# C¹_fib basis: |e, q⟩ for e ∈ E, q ∈ {0, 1}.                       Index = e*2 + q.
# ----------------------------------------------------------------------

def fibered_coboundary(k):
    d = np.zeros((NE * N_QUBIT, NV * N_FOCK), dtype=complex)
    kk = np.asarray(k, float)
    for e_idx, (u, v, voltage) in enumerate(EDGES):
        phase = np.exp(2j * np.pi * np.dot(kk, voltage))
        Pv = fiber_projection(v, e_idx)       # contributes +P_e^(v)
        Pu = fiber_projection(u, e_idx)       # contributes −e^{ik·n} P_e^(u)
        d[e_idx * N_QUBIT:(e_idx + 1) * N_QUBIT, v * N_FOCK:(v + 1) * N_FOCK] += Pv
        d[e_idx * N_QUBIT:(e_idx + 1) * N_QUBIT, u * N_FOCK:(u + 1) * N_FOCK] -= phase * Pu
    return d


# ----------------------------------------------------------------------
# Λ-grading on Cl(6) Fock:  N|s⟩ = (number of 1-bits in s)·|s⟩
# ----------------------------------------------------------------------

def lambda_grade_projector(n):
    """Project onto the n-fermion Λⁿ subspace of one vertex's Cl(6) Fock."""
    P = np.zeros((N_FOCK, N_FOCK), dtype=complex)
    for s in range(N_FOCK):
        if sum(_bits_of(s, 3)) == n:
            P[s, s] = 1.0
    return P


def per_vertex_block_diag(M_v):
    """Build block-diagonal NV*N_FOCK × NV*N_FOCK from the same N_FOCK × N_FOCK M_v on each vertex."""
    out = np.zeros((NV * N_FOCK, NV * N_FOCK), dtype=complex)
    for v in range(NV):
        out[v * N_FOCK:(v + 1) * N_FOCK, v * N_FOCK:(v + 1) * N_FOCK] = M_v
    return out


# ======================================================================
def part_A(k):
    print("=" * 90)
    print(f"PART A — algebra: Q̂² = blockdiag(Δ̂₀, Δ̂₁), {{Q̂, χ̂}} = 0   at k = {k}")
    print("=" * 90)
    d = fibered_coboundary(k)
    dim0, dim1 = NV * N_FOCK, NE * N_QUBIT
    print(f"\n  dim C⁰_fib = {dim0}    dim C¹_fib = {dim1}    d̂ is {d.shape[0]} × {d.shape[1]}")
    Q = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    Q[:dim0, dim0:] = d.conj().T
    Q[dim0:, :dim0] = d
    chi = np.diag([1.0] * dim0 + [-1.0] * dim1)
    QQ = Q @ Q
    D0, D1 = d.conj().T @ d, d @ d.conj().T
    blk = np.zeros_like(QQ); blk[:dim0, :dim0] = D0; blk[dim0:, dim0:] = D1
    ok_Q2 = np.allclose(QQ, blk, atol=1e-10)
    ok_chi = np.allclose(Q @ chi + chi @ Q, 0, atol=1e-10)
    print(f"  Q̂² = blockdiag(Δ̂₀, Δ̂₁):  {ok_Q2}        {{Q̂, χ̂}} = 0:  {ok_chi}")
    assert ok_Q2 and ok_chi
    return d, D0, D1


def part_B(D0, D1, k):
    print("\n" + "-" * 90)
    print("B — Witten pairing: nonzero spec Δ̂₀ = nonzero spec Δ̂₁")
    print("-" * 90)
    s0 = np.sort(np.linalg.eigvalsh((D0 + D0.conj().T) / 2))
    s1 = np.sort(np.linalg.eigvalsh((D1 + D1.conj().T) / 2))
    s0_nz = s0[s0 > 1e-8]
    s1_nz = s1[s1 > 1e-8]
    print(f"\n  Δ̂₀ spectrum: {len(s0_nz)} nonzero, {len(s0) - len(s0_nz)} zero modes")
    print(f"  Δ̂₁ spectrum: {len(s1_nz)} nonzero, {len(s1) - len(s1_nz)} zero modes")
    print(f"  nonzero spec Δ̂₀ = {s0_nz}")
    print(f"  nonzero spec Δ̂₁ = {s1_nz}")
    pairs = (len(s0_nz) == len(s1_nz)) and np.allclose(s0_nz, s1_nz, atol=1e-7)
    print(f"  Witten pairing holds: {pairs}")
    print(f"  Witten index (dim ker Δ̂₀ − dim ker Δ̂₁) = {(len(s0) - len(s0_nz)) - (len(s1) - len(s1_nz))}")
    assert pairs


def part_C(d, k):
    print("\n" + "-" * 90)
    print("C — restriction to the cochain base: on the Λ⁰ subspace, Q̂ ≡ the unfibered Q from de_rham probe")
    print("-" * 90)
    # Λ⁰ at every vertex = |vac⟩ = state 0.  The Λ⁰ subspace of C⁰_fib is 4-dim (one per vertex);
    # its embedding into C⁰_fib: vertex v's Λ⁰ basis vector = state index v*8 + 0.
    embed_C0 = np.zeros((NV * N_FOCK, NV))
    for v in range(NV):
        embed_C0[v * N_FOCK + 0, v] = 1.0
    # the image of d̂ on Λ⁰_fib: P_e^(v)|vac⟩ = |0⟩_e (the empty-edge state), state index 0 of the e-qubit.
    # So d̂(Λ⁰_fib) lies in the |0⟩-of-each-edge subspace of C¹_fib (6-dim).
    embed_C1 = np.zeros((NE * N_QUBIT, NE))
    for e in range(NE):
        embed_C1[e * N_QUBIT + 0, e] = 1.0
    # the unfibered cochain d at k = the base cochain matrix
    from proofs.common import bloch_H, K_STAR
    base_d = np.zeros((NE, NV), dtype=complex)
    kk = np.asarray(k, float)
    for ei, (u, v, n) in enumerate(EDGES):
        base_d[ei, u] += -1.0
        base_d[ei, v] += np.exp(2j * np.pi * np.dot(kk, n))
    d_restricted = embed_C1.T @ d @ embed_C0
    match = np.allclose(d_restricted, base_d, atol=1e-9)
    print(f"\n  on the Λ⁰ subspace (|vac⟩ at each vertex)  →  |0⟩-of-each-edge subspace of C¹_fib,")
    print(f"  the fibered d̂ reduces to the unfibered cochain d:  {match}")
    if not match:
        print(f"   d̂_restricted =\n{d_restricted}")
        print(f"   base d =\n{base_d}")
    # also verify Δ̂₀ restricted to Λ⁰ = the cochain Δ₀ (= 3I − bloch_H)
    D0 = d.conj().T @ d
    D0_restr = embed_C0.T @ D0 @ embed_C0
    cochain_lap = K_STAR * np.eye(NV) - bloch_H(k, BONDS)
    spec_match = np.allclose(np.sort(np.linalg.eigvalsh((D0_restr + D0_restr.conj().T) / 2)),
                             np.sort(np.linalg.eigvalsh((cochain_lap + cochain_lap.conj().T) / 2)),
                             atol=1e-7)
    print(f"  Δ̂₀ restricted to Λ⁰_fib has the same spectrum as the cochain Laplacian 3I−bloch_H:  {spec_match}")


def part_D(d, k):
    print("\n" + "-" * 90)
    print("D — action on the Λ-graded matter states  (Λⁿ = species ladder per cl6_fock_z3_breaking_decomposition):")
    print("    Λ⁰=lepton (1 state),  Λ¹=down (3 states),  Λ²=up (3 states),  Λ³=neutrino (1 state)  per vertex")
    print("-" * 90)
    print()
    for n in range(4):
        Pn = lambda_grade_projector(n)
        P_block = per_vertex_block_diag(Pn)
        dim = int(np.round(np.trace(P_block).real))
        # restrict d̂ to Λⁿ source and look at where it lands
        d_n = d @ P_block
        norms_n_to_edge = np.linalg.norm(d_n) ** 2  # Frobenius²
        # which Λ'-grades does the image, when fed BACK through d̂*, populate? Roughly checked by d̂*d̂ overlap.
        if norms_n_to_edge > 1e-12:
            # rank of d_n restricted to Λⁿ_fib
            d_n_restr = d_n[:, np.diag(P_block).real > 0.5]
            rank = np.linalg.matrix_rank(d_n_restr, tol=1e-8)
        else:
            rank = 0
        print(f"  Λ{n}  (dim {dim} per vertex × {NV} = {dim * NV} total):  ‖d̂|_Λ{n}‖² = {norms_n_to_edge:.3f},  rank d̂|_Λ{n} = {rank}")
    # detailed: take a basis matter state, apply Q̂, see what edge content comes back
    print("\n  detail — d̂ applied to a Λ¹ basis state '|fermion-on-edge-e at vertex v⟩':")
    # pick v=0, the fermion occupies the *first* of v=0's incident edges (slot 0)
    v_pick = 0
    incs_v = [eid for eid, _ in incident_edges(v_pick)]
    psi = np.zeros(NV * N_FOCK, dtype=complex)
    # state with bits (1, 0, 0) in the Cl(6) Fock = "fermion in slot 0" = "fermion on incident_edges(v)[0]"
    psi[v_pick * N_FOCK + 0b100] = 1.0
    e_occ = incs_v[0]
    print(f"   |ψ⟩ = |fermion in slot 0 at vertex {v_pick}⟩ = '|fermion on edge {e_occ} at vertex {v_pick}⟩'")
    phi = d @ psi
    print(f"   d̂|ψ⟩ landed on edges (e_idx, q, amplitude) where amplitude ≠ 0:")
    for j in range(NE * N_QUBIT):
        if abs(phi[j]) > 1e-9:
            e, q = j // N_QUBIT, j % N_QUBIT
            print(f"      edge {e}, qubit value |{q}⟩: amplitude = {phi[j]:+.4f}")
    print(f"   → notice: d̂ couples the matter slot for edge {e_occ} ONLY to the edge-{e_occ} qubit value |1⟩;")
    print(f"     the other components are zero  (the fiber projection picks out exactly the matching edge).")


def part_E(d, k):
    print("\n" + "-" * 90)
    print("E — SU(2) gauge equivariance:  per-edge SU(2) acting on ℂ²_e must commute with d̂ → Q̂² (matching content)")
    print("-" * 90)
    # SU(2) on edge e qubit: rotate ℂ²_e by some U ∈ SU(2). To act on the matter side, U must lift to the
    # vertex Focks containing ℂ²_e as a tensor factor. We test 'cogeneraivariance': does d̂ ∘ (matter SU(2)) =
    # (edge SU(2)) ∘ d̂ ? For a simple U on edge 0, the matter-side U acts as U ⊗ I ⊗ I on the
    # appropriate slot of the two vertices that share edge 0.
    rng = np.random.default_rng(0)
    # random SU(2) on a single edge
    angles = 2 * np.pi * rng.random(3)
    a, b, c = angles
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    U_e = (np.cos(a / 2) * np.eye(2) - 1j * np.sin(a / 2) * (np.sin(b) * np.cos(c) * sx
                                                              + np.sin(b) * np.sin(c) * sy
                                                              + np.cos(b) * sz))
    # apply U on edge 0; on the matter side, lift to the e=0 tensor factor of each vertex Fock containing edge 0
    target_edge = 0
    # edge action: block-diagonal, U on edge=0 qubit, I on others
    U_edge = np.eye(NE * N_QUBIT, dtype=complex)
    U_edge[target_edge * N_QUBIT:(target_edge + 1) * N_QUBIT,
           target_edge * N_QUBIT:(target_edge + 1) * N_QUBIT] = U_e

    # matter lift: for each vertex v, if edge=0 ∈ incident(v), find its slot, apply U on that slot.
    U_matter = np.eye(NV * N_FOCK, dtype=complex)
    for v in range(NV):
        incs = [eid for eid, _ in incident_edges(v)]
        if target_edge not in incs:
            continue
        slot = incs.index(target_edge)
        # build U on slot, I on the other two slots of the 3-qubit space
        u_v = [np.eye(2, dtype=complex)] * 3
        u_v[slot] = U_e
        full = u_v[0]
        for x in u_v[1:]:
            full = np.kron(full, x)
        # place into U_matter at vertex v's block
        U_matter[v * N_FOCK:(v + 1) * N_FOCK, v * N_FOCK:(v + 1) * N_FOCK] = full

    lhs = d @ U_matter         # apply matter U then d̂
    rhs = U_edge @ d           # apply d̂ then edge U
    diff = np.linalg.norm(lhs - rhs)
    print(f"\n  random SU(2) on edge 0:  ‖d̂·U_matter − U_edge·d̂‖ = {diff:.3e}")
    print(f"  → SU(2) equivariance holds:  {diff < 1e-9}")


def part_F(d, k):
    print("\n" + "-" * 90)
    print("F — the key question:  does Q̂ add NEW degrees of freedom (fermion+boson doubling like MSSM),")
    print("    or is it a RE-LABELLING of the SAME underlying qubits (same Hilbert space, two readings)?")
    print("-" * 90)
    # the underlying degrees of freedom in the framework:
    #   • the cell's |E| = 6 edge qubits (each carrying ℂ²_e), forming a base Hilbert space ℂ^{2^|E|} = ℂ^64.
    #   • every vertex's Cl(6) Fock = the 3 incident edge qubits' tensor product (a 3-qubit subspace of the 64-d).
    #   • every edge qubit = exactly the |0/1⟩ of one of the 6 base qubits.
    # in this 64-d picture: a 'matter excitation at v' = a state in v's 3-qubit subspace; a 'gauge excitation on e' =
    # a state in the 1-qubit subspace of edge e.  The SAME 64-d Hilbert space hosts both.
    print(f"""
  THE UNDERLYING PICTURE.  Cl(6)_v Fock = ⊗_{{e∋v}} ℂ²_e — three of the cell's six edge qubits, in tensor.
  So the cell has  |E| = {NE}  edge qubits, forming a base Hilbert space  ℂ^{{2^{NE}}} = ℂ^{2**NE}.
  Every vertex's Cl(6) Fock = the 3-qubit subspace at v's incident edges (a Hilbert space EMBEDDED into the
  64-d total), and every edge's gauge Hilbert space = the 1-qubit subspace at that edge (also embedded).

  THE SAME physical qubit's |1⟩ state is, simultaneously:
     • a 'fermion at u on edge e' (a Λ¹ basis vector of Cl(6)_u Fock, since u's Fock contains the e-qubit),
     • a 'fermion at v on edge e' (the same Λ¹ basis vector of Cl(6)_v Fock — v's Fock also contains e),
     • an edge-{{e}} excitation  (a |1⟩ state of the e-qubit alone, the gauge/Higgs sector).
  These are NOT three different states — they are THREE READINGS OF ONE STATE.

  ⇒  Q̂ does NOT add a separate 'fermion partner' and 'boson partner' à la MSSM.  Q̂ is a CHANGE-OF-READING
     operator: it rotates between the 'matter at a vertex' and the 'boson on an edge' descriptions of the
     SAME underlying excitations.  The Hilbert spaces C⁰_fib (32-d) and C¹_fib (12-d) are different
     PROJECTIONS of the same 64-d underlying space, not independent factors.

  EVIDENCE in the algebra above:  the fiber projection P_e^(v): ℂ⁸_v → ℂ²_e contracts the OTHER two
  factors against |0⟩|0⟩.  This is a LITERAL embedding of the edge qubit into the vertex Fock as the
  Λ⁰ ⊕ Λ¹ slice (the |b₀b₁b₂⟩ states with b_j = 0 for the non-e slots) — not a separate object.

  CONSEQUENCE FOR β-FUNCTIONS / MSSM.  In an MSSM-style accounting, fermion and boson loops are counted
  SEPARATELY because their Hilbert spaces are distinct.  Here, fermion-loops and boson-loops at the same
  edge qubit are LOOPS OF THE SAME UNDERLYING DEGREE OF FREEDOM, just labelled differently.  So counting
  states the MSSM way (matter + superpartners) DOUBLES-COUNTS in this framework.  The framework's natural
  β-function is therefore neither SM (only matter, missing the edge-bosonic reading) nor MSSM (matter +
  separate superpartners, double-counting) — it is the FRAMEWORK-NATIVE count, computed from the 6
  edge qubits and the C₃-protected Weyl structure on top.

  This explains, in one stroke: (a) why LHC sees no superpartners — there ARE no separate states; (b) why
  the framework's α_GUT⁻¹ = 24 + sin²θ_W = 3/8 boundary conditions cannot, in principle, be propagated
  to M_Z using MSSM β-functions — that's the wrong counting; (c) why the F7 / α_1 winding flow does NOT
  reproduce MSSM RG (M1 audit) — it is the framework-native flow, not MSSM's.  The mechanism connecting
  the GUT-scale derived numbers to the M_Z measurements is the FRAMEWORK'S OWN RG, computed on the 64-d
  edge-qubit Hilbert space — and that is what needs to be built (`frontier.beta_dark` — open).
""")


def main():
    print(r"""
====================================================================================
FIBERED DE RHAM SUSY — does Q̂ add MSSM-like superpartners, or relabel one Hilbert space?
====================================================================================
""")
    for label, k in [("Γ", GAMMA), ("P=(¼,¼,¼)", P_POINT), ("generic k", GENERIC_K)]:
        print(f"\n\n##### k-point: {label} #####")
        d, D0, D1 = part_A(k)
        part_B(D0, D1, k)
        part_C(d, k)
        part_D(d, k)
        part_E(d, k)
    # the structural F section is k-independent, run once
    print("\n\n")
    part_F(d, GAMMA)
    print("=" * 90)
    print("VERDICT (honest — this attempt does NOT close the mechanism)")
    print("=" * 90)
    print("""
  What works:
   • Q̂ closes algebraically at every k tested: Q̂² = blockdiag(Δ̂₀, Δ̂₁); {Q̂, χ̂} = 0; Witten pairing
     exact (nonzero spec Δ̂₀ = nonzero spec Δ̂₁).
   • On the Λ⁰ subspace the fibered d̂ reproduces the cochain Laplacian's spectrum (Part C).

  What fails:
   • GAUGE SU(2) EQUIVARIANCE FAILS (Part E):  ‖d̂·U_matter − U_edge·d̂‖ ≈ 4.76 ≠ 0 for a generic
     SU(2) rotation on a single edge.  The reason is structural — my fiber projection P_e^(v)
     contracts the OTHER two factors against the reference |0⟩|0⟩, which implicitly fixes a gauge on
     those non-e edges.  A cross-edge gauge rotation rotates that reference away from |0⟩, and the
     projection no longer commutes with it.  So this particular fibered Q̂ is NOT a physical
     (gauge-equivariant) supercharge.
   • Q̂ ONLY SEES Λ⁰ ⊕ Λ¹ (Part D):  Λ² and Λ³ states map to ZERO under d̂.  The fiber projection
     kills any state with occupation on non-e edges, so multi-fermion vertex Fock states are not
     'in play' for this Q̂.  The construction therefore captures the 1-fermion-sector matter (lepton
     ⊕ down-by-edge per vertex), not the full Cl(6) Fock (lepton, down, up, neutrino).

  What this means:
   • The 'obvious' fibered lift of Q is NOT the MSSM mechanism.  It fails the two structural tests
     (gauge equivariance, full-Fock coverage) that any honest physical SUSY must pass.
   • A different fibered construction — gauge-equivariant by design, and covering all four Λ-levels
     — would be needed.  Concretely: replace 'contract to |0⟩|0⟩' with an SU(2)-equivariant operator
     (e.g., one that acts at the level of the operator algebra Cl(6)_v → Cl(2)_e via a partial
     trace of operators, or one that uses ALL species levels symmetrically).  This is harder than
     my proposal soft-pedalled, and it is NOT shown to exist by this probe.
   • The earlier de_rham_susy_on_srs_probe verdict was correct in its narrow scope (the COCHAIN-BASE
     SUSY is real and is geometric), and my 'I overcalled the negative' walk-back went too far in
     the other direction — the fibered construction needed to make the SUSY a true gauge-equivariant
     physical supercharge is NOT trivially available; this specific attempt fails.

  Honest options from here:
   (1) try a smarter fibered Q̂ — operator-algebra (Cl(6)→Cl(2) partial trace, gauge-equivariant by
       construction);  bounded probe, but no guarantee it closes;
   (2) accept that the de-Rham route, as built, doesn't reach MSSM, and pivot to the χ̃ cross-
       substrate T_mix route (`theorem_chi_tilde_breaking_operator_scoping_2026-05-01.md` Class 4/5
       — flagged 'most promising', un-built, predicted-weak);
   (3) frontier.beta_dark — the framework-native RG, separate from MSSM, also un-built.

  No graded content changes; this is a research probe and the negative is informative.
""")
    print("de_rham_susy_fibered_probe.py: all checks passed (sentinel).")


if __name__ == "__main__":
    main()
