#!/usr/bin/env python3
"""
de_rham_susy_on_srs_probe.py — Is there a Witten–Hodge SUSY hiding inside B(srs)?

MOTIVATION
==========
The framework's "MSSM" cannot be found by enumerating substrate nets / algebras
(ADOPTED-MSSM-Sb; `simulator_skeleton/frontier.py` gap `mssm_as_adoption`).  But
the field content sorts cleanly by graph degree:

  matter fermions   → Cl(6) Fock AT VERTICES        = 0-cochains of the srs cell
  gauge SU(2)_L×_R  → Cl(0,2)≅ℍ ON EDGES            = 1-cochains
  Higgs             → Cl(2)     ON EDGES            = 1-cochains

On a graph the Witten supercharge Q = d + d* maps 0-cochains ↔ 1-cochains with
Q² = the Hodge Laplacian, and the Bloch graph Laplacian Δ₀(k) = D − A(k)
IS the operator the framework calls B(srs) (up to the trivial shift 3I − B).
So the question: does B(srs) carry a built-in N=1 SUSY-QM whose ℤ₂-grading is the
cochain degree — i.e. does the framework's matter sector sit in a supermultiplet
with its gauge/Higgs sector?  And if so, is that the MSSM entry point or merely a
"geometric" SUSY that does not flip statistics?

WHAT THIS PROBE BUILDS / CHECKS
===============================
Part 1 (rigorous, sentinel asserts):
  • the Bloch-twisted cochain complex of the srs primitive cell (K₄ + ℤ³ voltages
    from `proofs/common.py`): C⁰=ℂ⁴ (vertices), C¹=ℂ⁶ (edges), coboundary d(k);
  • Δ₀(k) = d(k)†d(k)  — verified to have spectrum  3 − spec(bloch_H(k));
  • Δ₁(k) = d(k)d(k)†; supercharge Q(k) = [[0, d†],[d, 0]] on C⁰⊕C¹ (10-dim);
  • Q(k)² = blockdiag(Δ₀, Δ₁);  {Q(k), χ} = 0  with  χ = diag(+I₄, −I₆);
  • Witten pairing: nonzero spec(Δ₀) = nonzero spec(Δ₁) (multiplicities match);
  • Witten index  ind Q = dim ker Q|_{C⁰} − dim ker Q|_{C¹} = χ(K₄) = |V|−|E| = −2,
    and it is k-INDEPENDENT (topological);
  • harmonic content: Γ → (1 scalar, 3 vectors); P=(¼,¼,¼) → computed; generic k →
    (0, 2).  The "3" at Γ = first Betti number of the cell = the cycle-space rank
    the framework already uses for n_gen.

Part 2 (exploratory, observations + checks, no claim of closure):
  • the vertex Fock is Λ•(ℂ³) = Cl(6,ℂ) Fock ≅ Cl(2,ℂ)^⊗3, the 3 tensor factors
    labelled by the vertex's 3 incident edges (verify the algebra iso + the
    {γ_a,γ_b}=2δ_ab relations + the Λ-grading dims 1,3,3,1);
  • each edge qubit ℂ²_e is a tensor factor SHARED by the two incident vertices'
    Focks → the framework's matter (degree-0) and gauge (degree-1) sectors are
    literally the two halves of this complex, glued along the shared edge qubits;
  • the framework carries THREE ℤ-/ℤ₂-gradings on the matter sector: cochain degree
    (vertex vs edge), Λ-degree (= the lepton/down/up/ν species ladder n=0..3, per
    `cl6_fock_z3_breaking_decomposition.py`), and γ₇ chirality.  Check whether Q (or
    any of these, or a product) flips boson↔fermion statistics — it does not, for
    the structural reason that everything lives in a Clifford module.

VERDICT (printed at the end): geometric/Hodge SUSY YES (and now packaged);
physical/MSSM SUSY NO — and the cochain-language statement of *why* is sharper than
where the χ̃ thread (`theorem_chi_tilde_breaking_operator_scoping_2026-05-01.md`,
`srs_z_chi_*`) currently sits.

This file is NOT a theorem and changes no graded content.  It is a structural probe.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, bloch_H, K_STAR, N_ATOMS  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=120)

TOL = 1e-9
GAMMA = (0.0, 0.0, 0.0)
P_POINT = (0.25, 0.25, 0.25)
GENERIC_K = (0.137, 0.291, 0.453)   # an irrational-ish generic point


# ======================================================================
# The srs primitive cell as a voltage graph (K₄ + ℤ³ voltages)
# ======================================================================

def build_cell_graph():
    """Return (V, edges) where V = list(range(4)) and edges = list of
    (u, v, voltage) with voltage ∈ ℤ³ — one oriented representative per
    undirected edge of the srs primitive cell.

    `find_bonds()` gives 12 directed bonds (4 atoms × 3 NN).  Each undirected
    edge {u,v} appears as (u,v,+c) and (v,u,−c); we keep the representative with
    the lexicographically smaller (u,v) (and for the rare u==v / parallel case,
    keep distinct voltages).
    """
    bonds = find_bonds()
    seen = {}
    for u, v, c in bonds:
        c = tuple(int(x) for x in c)
        key = (min(u, v), max(u, v), tuple(sorted((c, tuple(-x for x in c)))))
        if key in seen:
            continue
        # orient u→v with u ≤ v; flip voltage if we had to swap
        if u <= v:
            seen[key] = (u, v, c)
        else:
            seen[key] = (v, u, tuple(-x for x in c))
    edges = sorted(seen.values())
    V = list(range(N_ATOMS))
    return V, edges


def coboundary(k_frac, V, edges):
    """d(k): C⁰(=ℂ^|V|) → C¹(=ℂ^|E|).  Row e=(u→v,n): −1 at col u, e^{2πi k·n} at col v."""
    nE, nV = len(edges), len(V)
    d = np.zeros((nE, nV), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for e_idx, (u, v, n) in enumerate(edges):
        d[e_idx, u] += -1.0
        d[e_idx, v] += np.exp(2j * np.pi * np.dot(k, n))
    return d


def hodge_laplacians(d):
    return d.conj().T @ d, d @ d.conj().T   # (Δ₀ on C⁰, Δ₁ on C¹)


def supercharge(d):
    """Q on C⁰⊕C¹:  [[0, d†],[d, 0]].  χ = diag(+I_{|V|}, −I_{|E|})."""
    nE, nV = d.shape
    Q = np.zeros((nV + nE, nV + nE), dtype=complex)
    Q[:nV, nV:] = d.conj().T
    Q[nV:, :nV] = d
    chi = np.diag(np.array([1.0] * nV + [-1.0] * nE))
    return Q, chi


def nullity(M, tol=1e-7):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s < tol))


def sorted_pos_eigs(H, tol=1e-7):
    w = np.linalg.eigvalsh((H + H.conj().T) / 2)
    return np.sort(w[w > tol])


# ======================================================================
# PART 1 — the graph-level Witten–Hodge complex
# ======================================================================

def part1():
    print("=" * 78)
    print("PART 1 — Witten–Hodge SUSY-QM on the srs primitive cell  Q = d + d*")
    print("=" * 78)

    V, edges = build_cell_graph()
    nV, nE = len(V), len(edges)
    print(f"\n  cell graph: |V| = {nV}, |E| = {nE}")
    pairs = {(u, v) for u, v, _ in edges}
    is_K4 = (nV == 4 and nE == 6 and len(pairs) == 6)
    print(f"  edges (u, v, voltage):")
    for u, v, n in edges:
        print(f"    {u}—{v}  voltage {n}")
    assert is_K4, "expected the simple K₄ quotient (4 vertices, 6 distinct edges)"
    print(f"  ⇒ the quotient is the simple graph K₄ (with ℤ³ voltages).  Euler char χ = |V|−|E| = {nV-nE}.")

    for name, kf in [("Γ = (0,0,0)", GAMMA), ("P = (¼,¼,¼)", P_POINT), ("generic k", GENERIC_K)]:
        print("\n  " + "-" * 72)
        print(f"  k-point: {name}")
        print("  " + "-" * 72)
        d = coboundary(kf, V, edges)
        D0, D1 = hodge_laplacians(d)
        Q, chi = supercharge(d)

        # (a) Δ₀(k) is the framework's Bloch Laplacian 3I − B(srs)(k)  (spectrum-level, convention-free)
        H = bloch_H(kf, find_bonds())
        spec_D0 = np.sort(np.linalg.eigvalsh((D0 + D0.conj().T) / 2))
        spec_3mH = np.sort(K_STAR - np.linalg.eigvalsh((H + H.conj().T) / 2))
        ok_bloch = np.allclose(spec_D0, spec_3mH, atol=1e-7)
        print(f"    spec Δ₀(k)            = {spec_D0}")
        print(f"    spec(k* − bloch_H(k)) = {spec_3mH}   match: {ok_bloch}")
        assert ok_bloch, "Δ₀(k) must be the Bloch graph Laplacian 3I − B(srs)(k)"

        # (b) Q² = blockdiag(Δ₀, Δ₁);  {Q, χ} = 0
        QQ = Q @ Q
        blk = np.zeros_like(QQ); blk[:nV, :nV] = D0; blk[nV:, nV:] = D1
        ok_Q2 = np.allclose(QQ, blk, atol=1e-9)
        ok_grade = np.allclose(Q @ chi + chi @ Q, 0, atol=1e-9)
        print(f"    Q² = blockdiag(Δ₀, Δ₁): {ok_Q2}     {{Q, χ}} = 0: {ok_grade}")
        assert ok_Q2 and ok_grade

        # (c) Witten pairing of the nonzero spectrum
        pos0, pos1 = sorted_pos_eigs(D0), sorted_pos_eigs(D1)
        ok_pair = (len(pos0) == len(pos1)) and np.allclose(pos0, pos1, atol=1e-7)
        print(f"    nonzero spec Δ₀ = {pos0}")
        print(f"    nonzero spec Δ₁ = {pos1}      Witten pairing holds: {ok_pair}")
        assert ok_pair

        # (d) Witten index = χ(graph), k-independent
        n0, n1 = nullity(d.conj().T @ d), nullity(d @ d.conj().T)
        # ker Q|_{C⁰} = ker Δ₀ = ker d ;  ker Q|_{C¹} = ker Δ₁ = ker d†
        h0, h1 = nV - np.linalg.matrix_rank(d, tol=1e-7), nE - np.linalg.matrix_rank(d, tol=1e-7)
        print(f"    harmonic content:  dim H⁰ (unpaired 'bosons')  = {h0}")
        print(f"                       dim H¹ (unpaired 'fermions') = {h1}")
        print(f"    Witten index  ind Q = dim H⁰ − dim H¹ = {h0 - h1}   (Euler char χ(K₄) = {nV - nE})")
        assert h0 - h1 == nV - nE, "Witten index must equal the Euler characteristic (topological)"
        if kf == GAMMA:
            betti1 = nE - nV + 1
            print(f"    [Γ] dim H¹ = first Betti number of the cell = |E|−|V|+1 = {betti1}"
                  f"  — the cycle-space rank the framework uses for n_gen = 3.")
            assert h1 == betti1 == 3
        if kf == P_POINT:
            print(f"    [P] Δ₀(P) eigenvalues are k* ∓ √k* = 3 ∓ √3 ≈ {3 - np.sqrt(3):.4f}, {3 + np.sqrt(3):.4f}"
                  f"  (because bloch_H(P)² = k*·I — the 'complex structure J = H/√k*' point).")

    print("\n  PART 1 verdict:  B(srs) is the degree-0 piece of a genuine Witten–Hodge")
    print("  N=1 SUSY-QM on the srs cell's cochain complex.  Q = d + d* swaps the")
    print("  vertex sector (degree 0) with the edge sector (degree 1); Q² is the Hodge")
    print("  Laplacian; the supersymmetry breaking is TOPOLOGICAL — the unpaired states")
    print("  are χ(K₄) = −2 worth of harmonics: 1 scalar + 3 vectors at Γ, 0 + 2 generically.")
    return V, edges


# ======================================================================
# PART 2 — the fiber structure: vertices = Λ•(ℂ³) = Cl(6) Fock, edges = Cl(0,2)
# ======================================================================

def _cl2_gens():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def _kron(*mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def part2(V, edges):
    print("\n" + "=" * 78)
    print("PART 2 — fiber structure: vertex Fock = Λ•(ℂ³) ≅ Cl(6,ℂ) ≅ Cl(2,ℂ)^⊗3,")
    print("         edge fiber = Cl(0,2) ≅ ℍ acting on the shared edge qubit ℂ²")
    print("=" * 78)

    sx, sy, sz = _cl2_gens()
    I2 = np.eye(2, dtype=complex)
    N1 = (I2 - sz) / 2  # number operator on one Cl(2) factor

    # Cl(6,ℂ) generators by Jordan–Wigner over 3 qubit factors  (the 3 incident edges)
    g = []
    for j in range(3):
        left = [sz] * j
        right = [I2] * (2 - j)
        g.append(_kron(*left, sx, *right))   # γ_{2j+1}
        g.append(_kron(*left, sy, *right))   # γ_{2j+2}
    # {γ_a, γ_b} = 2 δ_ab I_8
    ok_cliff = all(
        np.allclose(g[a] @ g[b] + g[b] @ g[a], (2.0 if a == b else 0.0) * np.eye(8), atol=1e-12)
        for a in range(6) for b in range(6))
    print(f"\n  Cl(6,ℂ) from JW over 3 edge-qubits:  {{γ_a, γ_b}} = 2δ_ab I₈  →  {ok_cliff}")
    assert ok_cliff

    # Λ-grading: N = N_1 ⊗ I ⊗ I  +  I ⊗ N_1 ⊗ I  +  I ⊗ I ⊗ N_1
    N = _kron(N1, I2, I2) + _kron(I2, N1, I2) + _kron(I2, I2, N1)
    levels = {}
    for n in np.round(np.linalg.eigvalsh(N)).astype(int):
        levels[n] = levels.get(n, 0) + 1
    print(f"  Λ-grading (exterior degree n = #occupied edge-qubits):  dims by level = "
          f"{[levels[n] for n in sorted(levels)]}  (expected 1,3,3,1 = binom(3,n))")
    assert [levels[n] for n in sorted(levels)] == [1, 3, 3, 1]
    print( "  n = 0,1,2,3  ↔  lepton, down, up, neutrino species ladder"
           "  (cl6_fock_z3_breaking_decomposition.py).")

    # γ₇ = γ₁γ₂γ₃γ₄γ₅γ₆ = (−1)^N  — chirality = exterior-degree parity, NOT statistics
    g7 = g[0] @ g[1] @ g[2] @ g[3] @ g[4] @ g[5]
    parity = _kron(sz, sz, sz)
    # γ₇ ∝ (−1)^F up to a phase; check it is ± the parity operator
    ratio = g7 @ np.linalg.inv(parity)
    ok_g7 = np.allclose(ratio, ratio[0, 0] * np.eye(8), atol=1e-9) and abs(abs(ratio[0, 0]) - 1) < 1e-9
    print(f"  γ₇ = γ₁…γ₆ ∝ (−1)^N (= ± diag of edge-qubit parities):  {ok_g7}"
          f"  → γ₇ grades Λ-degree parity, i.e. chirality, not boson/fermion statistics.")
    assert ok_g7

    # The shared-edge-qubit gluing: each edge appears as a tensor factor in BOTH endpoints' Focks.
    incident = {v: [i for i, (u, w, _) in enumerate(edges) if v in (u, w)] for v in V}
    print("\n  incident-edge lists (each vertex's Cl(6) = ⊗ over these 3 edge-qubits):")
    for v in V:
        print(f"    vertex {v}:  edges {incident[v]}  ⇒  ℂ²_{incident[v][0]} ⊗ ℂ²_{incident[v][1]} ⊗ ℂ²_{incident[v][2]}")
    shared_ok = all(len(incident[v]) == 3 for v in V) and all(
        sum(1 for v in V if e in incident[v]) == 2 for e in range(len(edges)))
    print(f"  every edge-qubit ℂ²_e is shared by EXACTLY its 2 endpoint vertices: {shared_ok}")
    assert shared_ok
    print( "  ⇒ the framework's matter sector (the vertex Focks) and its gauge+Higgs sector")
    print( "    (the edge qubits, via Cl(0,2)≅ℍ → SU(2)_L×SU(2)_R and Cl(2) → Higgs) are the")
    print( "    degree-0 and degree-1 halves of ONE complex, glued along the shared edge qubits.")
    print( "    Q = d + d* is the operator that moves between the two halves.")

    # Cl(0,2) ≅ ℍ on a single edge qubit ℂ²: e₁ = iσx, e₂ = iσy (both square to −I, anticommute);
    # i_ℍ = e₁, j_ℍ = e₂, k_ℍ = e₁e₂ = −iσz ;  i²=j²=k²=ijk=−I  (theorem_g2_edge_qubit_su2.md §4).
    iH, jH = 1j * sx, 1j * sy
    kH = iH @ jH
    ok_H = (all(np.allclose(m @ m, -I2, atol=1e-12) for m in (iH, jH, kH))
            and np.allclose(iH @ jH + jH @ iH, 0, atol=1e-12)
            and np.allclose(iH @ jH @ kH, -I2, atol=1e-12))
    print(f"\n  edge fiber Cl(0,2)≅ℍ on ℂ²_e:  i²=j²=k²=ijk=−I → {ok_H}  (SU(2)=Sp(1) acts on ℂ²_e)")
    assert ok_H

    print("\n  PART 2 verdict:  the matter↔gauge correspondence is the cochain-degree swap")
    print("  of Part 1, made concrete on the fibers.  But there is NO ℤ₂ here that flips")
    print("  boson↔fermion statistics: the matter sector is a Clifford module, and ALL THREE")
    print("  gradings it carries are degree-type — cochain degree (vertex vs edge), Λ-degree")
    print("  (the n=0..3 species ladder), γ₇ (Λ-parity = chirality).  A genuine super-")
    print("  symmetry à la MSSM would need a ℤ₂ acting TRANSVERSALLY to all three; the")
    print("  Clifford-module structure has no room for it.  (Same obstruction the χ̃ thread")
    print("  hit, now in cochain language: 'χ̃ ≡ γ₇ lifted through srs-z's cover' grades")
    print("  Λ-parity, hence chirality, hence not statistics.)")


# ======================================================================
def main():
    V, edges = part1()
    part2(V, edges)
    print("\n" + "=" * 78)
    print("OVERALL")
    print("=" * 78)
    print("""
  GEOMETRIC / HODGE SUSY:  YES — and now packaged.  B(srs) is literally the
    degree-0 block of Q² for the Witten supercharge Q = d + d* on the srs cell's
    cochain complex; the framework's matter (degree 0) and gauge+Higgs (degree 1)
    sectors are the two halves of that complex; the breaking is topological
    (unpaired harmonics = χ(K₄) = −2: a scalar + 3 vectors at Γ, where 3 = the
    cycle-space rank = n_gen; 0 + 2 generically).

  PHYSICAL / MSSM SUSY:    NO — for the now-sharpened reason: the matter sector's
    only ℤ₂/ℤ-gradings are degree-type (cochain degree, Λ-species degree, γ₇ =
    Λ-parity).  Q swaps degrees; it does NOT flip statistics.  An MSSM superpartner
    map needs a boson↔fermion ℤ₂ transverse to all three existing gradings, which a
    Clifford module cannot host — the same obstruction the χ̃ thread documented
    (`theorem_chi_tilde_breaking_operator_scoping_2026-05-01.md`, `srs_z_chi_*`),
    here recast: 'the would-be SUSY grading is forced to coincide with a degree
    grading, which is observably inert'.

  SO:  the "MSSM entry point the simulator was missing" is NOT a substrate-menu
    item — and it is also NOT a hidden reading.  The cochain reading EXISTS, it is
    the Witten–Hodge SUSY above, and it is geometric, not statistical.  This probe
    is the cleanest statement to date of *why* the entry point is absent; it does
    not open one.  Frontier gap `mssm_as_adoption` stands; ADOPTED-MSSM-Sb stands.
""")
    print("de_rham_susy_on_srs_probe.py: all checks passed (sentinel).")


if __name__ == "__main__":
    main()
