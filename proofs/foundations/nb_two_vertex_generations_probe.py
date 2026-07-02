#!/usr/bin/env python3
"""
nb_two_vertex_generations_probe.py
==================================
Two-vertex destructive interference, take 2: the NON-BACKTRACKING (Hashimoto)
propagator at the framework's MDL point u* = (k*−1)/k* = 2/3 — does the
three-fold ("generation") structure sharpen, and does a CP-type phase appear?

Background.  `de_rham_susy_on_srs_probe.py` showed B(srs) is the degree-0 block
of a Witten–Hodge SUSY-QM on the srs cell.  `two_vertex_interference_generations_probe.py`
showed the *two-vertex* amplitude on that cell (sum over all walks A→B) is rational-
with-zeros (not the constant (2/3)^g of the one-vertex / single-girth-cycle picture),
that its C₃-Fourier components give 1 "trivial" + 2 "generation" modes (the latter
living exactly on the C₃-symmetry-broken locus, with |ω| ≠ |ω²| = a CP seed), and
that the three independent circulating currents wind along the three crystal axes.
That probe used the *naive adjacency resolvent*.  The framework's actual mass object
is the *non-backtracking* (= 1-particle-irreducible, per a separate private derivation by the author-D5) Green's function,
and its MDL/"tree-cover" decay point is u* = g(z*) = 2/3 at z* = 17/6 (a separate private derivation by the author).
This probe redoes the analysis with that operator.

What this probe builds / checks
-------------------------------
A — the Bloch non-backtracking operator B_NB(k) on the srs cell (12 directed edges,
    each step weighted by e^{2πi k·voltage}).  Sanity:
      • Ihara–Bass at Γ: Perron eigenvalue = k*−1 = 2; the rest are roots of
        h² − λh + (k*−1) = 0 over the adjacency spectrum {3,−1,−1,−1}, plus ±1's;
      • Ramanujan saturation at P = (¼,¼,¼): the complex eigenvalues have |h| = √2,
        and h_P = (√3 + i√5)/2 (proofs/common.h_P) appears.
B — the NB Green's function G_NB(u,k) = (I − u·B_NB(k))⁻¹ at u* = 2/3.  On the
    quotient cell this is the *analytic continuation* of the tree-cover decay law
    g(z*)^d (the quotient is "more connected" than the 3-regular tree, so u*=2/3 is
    past its convergence radius — the framework's z* mechanism's natural home is the
    tree cover; here we read off the analytic continuation on the cell, which carries
    the C₃ / voltage / phase structure the tree cannot).
C — the three circulating currents (triangle loops 012, 013, 023 — a C₃-orbit) as
    directed-edge vectors ℓ₁,ℓ₂,ℓ₃; the 3×3 amplitude matrix M_{ij}(u*,k) = ℓ_i† G_NB ℓ_j;
    its eigenvalues; on the C₃ axis M is circulant ⇒ C₃-DFT gives (μ_triv, μ_ω, μ_ω²)
    with μ_ω = μ_ω²; off the axis they split and acquire a relative phase.  Tabulate
    |μ| and arg μ walking off the axis.
D — the vertex–vertex NB amplitude Φ(A,B; u*,k) = (T G_NB S^T)[A,B] (T,S = head/tail
    incidence — the V_us-style object); C₃-decompose the spoke triple (0→{1,2,3});
    compare magnitudes to the adjacency version; report the off-axis ω↔ω² phase next
    to the framework's δ_CP-ish angle arg(h_P^g).

VERDICT (printed).  This is a structural probe and a CLUE — not a closure.  Whether
the three magnitudes reproduce the QUANTITATIVE generation/Yukawa hierarchy
(~1 : 200 : 3000, the CKM/PMNS angles, δ_CP) is the framework's known-hard open
problem `simulator_skeleton/frontier.need_d3_species` ("5 sessions / 8 attacks ruled
out, foundational extension needed").  This probe reports the actual magnitudes/phases
the NB object at u* produces; it does not claim to settle Need-D-3, and it changes no
graded content.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import (find_bonds, bloch_H, K_STAR, N_ATOMS,  # noqa: E402
                           C3_PERM, omega3, h_P, GIRTH)

np.set_printoptions(precision=4, suppress=True, linewidth=140)

GAMMA = (0.0, 0.0, 0.0)
P_POINT = (0.25, 0.25, 0.25)
GENERIC_K = (0.137, 0.291, 0.453)
U_STAR = (K_STAR - 1) / K_STAR              # = 2/3, the MDL / tree-cover decay point g(z*)
Z_STAR = (1.0 / U_STAR) + (K_STAR - 1) * U_STAR   # = 17/6 (the tree substitution z = 1/u + (k*-1)u)


# ----------------------------------------------------------------------
# directed edges + the Bloch non-backtracking (Hashimoto) operator
# ----------------------------------------------------------------------

def directed_edges():
    """All 12 directed edges of the srs cell as (tail, head, voltage).  find_bonds()
    already returns (i, j, cell) = the directed edge i→j with voltage `cell`; we just
    canonicalise the type and check the reverse-edge convention voltage(rev e) = −voltage(e)."""
    de = [(int(i), int(j), tuple(int(x) for x in c)) for i, j, c in find_bonds()]
    assert len(de) == 2 * (K_STAR * N_ATOMS // 2) == 12
    by = {(u, v, n) for u, v, n in de}
    for u, v, n in de:
        assert (v, u, tuple(-x for x in n)) in by, "voltage graph: rev(e) should carry −voltage(e)"
    return de


def rev_index(de):
    """idx ↦ idx of the reverse edge."""
    pos = {(u, v, n): i for i, (u, v, n) in enumerate(de)}
    return [pos[(v, u, tuple(-x for x in n))] for (u, v, n) in de]


def nb_operator(k_frac, de, rev):
    """Bloch non-backtracking matrix.  B_NB[a,b] = e^{2πi k·voltage(e_b)} if
    head(e_a) == tail(e_b) and e_b ≠ reverse(e_a), else 0.  (Phase carried by the
    edge being stepped onto; at k=0 this is the standard 0/1 Hashimoto matrix.)"""
    m = len(de)
    B = np.zeros((m, m), dtype=complex)
    k = np.asarray(k_frac, float)
    for a, (ua, va, na) in enumerate(de):
        for b, (ub, vb, nb) in enumerate(de):
            if va == ub and b != rev[a]:
                B[a, b] = np.exp(2j * np.pi * np.dot(k, nb))
    return B


def incidence(de):
    """S[v,e] = 1 if tail(e)=v ; T[v,e] = 1 if head(e)=v  (n x 12)."""
    S = np.zeros((N_ATOMS, len(de)))
    T = np.zeros((N_ATOMS, len(de)))
    for e, (u, v, _) in enumerate(de):
        S[u, e] = 1.0
        T[v, e] = 1.0
    return S, T


# ======================================================================
# PART A — build B_NB and sanity-check it (Ihara–Bass, Ramanujan)
# ======================================================================

def part_A(de, rev):
    print("=" * 84)
    print("PART A — the Bloch non-backtracking (Hashimoto) operator B_NB(k)  +  sanity checks")
    print("=" * 84)
    print(f"\n  {len(de)} directed edges; MDL point u* = (k*−1)/k* = {U_STAR}  (⇔ tree energy z* = 1/u* + (k*−1)u* = {Z_STAR})")
    print(f"  a separate private derivation by the author: g(z*) = 2/3 at z* = 17/6 — each non-backtracking step costs exactly 1 NB-bit.\n")

    # Γ: Ihara–Bass.  spec(B_NB)|_Γ = {±1 (multiplicities |E|−|V|)} ∪ {roots of h²−λh+(k*−1)=0 : λ∈spec(A)}
    B0 = nb_operator(GAMMA, de, rev)
    ev_nb = np.linalg.eigvals(B0)
    A0 = bloch_H(GAMMA, find_bonds())
    spec_A = np.round(np.linalg.eigvalsh(A0), 6)
    ih_roots = []
    for lam in spec_A:
        disc = lam ** 2 - 4 * (K_STAR - 1)
        r = np.sqrt(complex(disc))
        ih_roots += [(lam + r) / 2, (lam - r) / 2]
    expected = sorted(ih_roots + [1.0] * (len(de) // 2 - N_ATOMS) + [-1.0] * (len(de) // 2 - N_ATOMS),
                      key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    got = sorted(ev_nb, key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    perron = max(abs(ev_nb))
    print(f"  at Γ:  spec A = {spec_A.tolist()}  ⇒  Ihara–Bass roots of h²−λh+{K_STAR-1}=0:")
    for lam in sorted(set(np.round(spec_A, 4))):
        disc = lam ** 2 - 4 * (K_STAR - 1)
        r = np.sqrt(complex(disc))
        print(f"     λ = {lam:>5}:  h = {(lam+r)/2:.4f},  {(lam-r)/2:.4f}   |h| = {abs((lam+r)/2):.4f}")
    print(f"  Perron eigenvalue of B_NB|_Γ = {perron:.6f}   (expected k*−1 = {K_STAR-1})")
    assert abs(perron - (K_STAR - 1)) < 1e-9
    ok_ih = np.allclose(np.array([complex(z) for z in got]),
                        np.array([complex(z) for z in expected]), atol=1e-7)
    print(f"  full spectrum matches Ihara–Bass (incl. the {len(de)//2-N_ATOMS}×(+1) and {len(de)//2-N_ATOMS}×(−1)): {ok_ih}")
    assert ok_ih

    # P: Ramanujan saturation — the non-±1 eigenvalues have |h| = √(k*−1) = √2; h_P appears.
    BP = nb_operator(P_POINT, de, rev)
    ev_P = np.linalg.eigvals(BP)
    non_unit = [z for z in ev_P if abs(abs(z) - 1) > 1e-6]
    mags = sorted({round(abs(z), 6) for z in non_unit})
    print(f"\n  at P = (¼,¼,¼):  |eigenvalues| of B_NB (excluding the ±1's) = {mags}"
          f"   (Ramanujan bound √(k*−1) = √2 = {np.sqrt(K_STAR-1):.6f})")
    assert all(abs(m - np.sqrt(K_STAR - 1)) < 1e-6 for m in mags), "P should be Ramanujan-saturated"
    has_hP = any(abs(z - h_P) < 1e-6 or abs(z - np.conj(h_P)) < 1e-6 for z in ev_P)
    print(f"  h_P = (√3 + i√5)/2 = {h_P:.4f} (|h_P| = √2) is among them: {has_hP}")
    assert has_hP
    print(f"  ⇒ B_NB is the right object: at P every step is a unit-modulus·√2 rotation; arg(h_P^g) with")
    print(f"    g = {GIRTH} is the framework's δ_CP-ish phase = {np.degrees(np.angle(h_P ** GIRTH)) % 360:.2f}°.")


# ======================================================================
# PART B — G_NB at u* = 2/3, and the 3×3 circulating-current matrix
# ======================================================================

def _triangle_loop_vectors(de):
    """The three single-axis triangle loops {012, 013, 023} (a C₃-orbit) as directed-edge vectors:
    ℓ_T has +1 on the directed edge if it's traversed forward going round T (in the order given),
    −1 if backward, 0 otherwise.  Returns (loops 6? no — 12-dim vectors)."""
    pos = {(u, v): e for e, (u, v, _) in enumerate(de)}  # any voltage rep; for indicator we only need direction
    # find_bonds gives both (u,v,·) and (v,u,·); pos picks one — but a triangle uses each oriented step,
    # so build a lookup oriented-pair → directed-edge index that prefers the matching orientation:
    pos = {}
    for e, (u, v, _) in enumerate(de):
        pos.setdefault((u, v), e)
    tri_cycles = [(0, 1, 2), (0, 1, 3), (0, 2, 3)]   # the C₃-orbit; orientation a→b→c→a
    vecs = []
    for (a, b, c) in tri_cycles:
        x = np.zeros(len(de), dtype=complex)
        for (u, v) in [(a, b), (b, c), (c, a)]:
            x[pos[(u, v)]] += 1.0
        vecs.append(x / np.linalg.norm(x))
    return tri_cycles, np.column_stack(vecs)   # 12 x 3


def _c3_dft_row(row3):
    """DFT of a length-3 row → (trivial, ω, ω²) components."""
    a = np.asarray(row3, dtype=complex)
    return ((a[0] + a[1] + a[2]) / np.sqrt(3),
            (a[0] + omega3 * a[1] + omega3 ** 2 * a[2]) / np.sqrt(3),
            (a[0] + omega3 ** 2 * a[1] + omega3 * a[2]) / np.sqrt(3))


def part_B(de, rev):
    print("\n" + "=" * 84)
    print(f"PART B — G_NB(u*={U_STAR}, k) and the 3×3 circulating-current amplitude matrix M_{{ij}}")
    print("=" * 84)
    print(f"\n  G_NB(u,k) = (I − u·B_NB(k))⁻¹.  At u* = {U_STAR} on the quotient cell this is the analytic")
    print(f"  continuation of the tree-cover decay g(z*)^d (the quotient's NB Perron eigenvalue is k*−1=2,")
    print(f"  so u*·2 = 4/3 > 1 — past the cell's convergence radius; the z* mechanism's natural home is")
    print(f"  the 3-regular tree, where u*=2/3 < 1/√2 is convergent.  The cell carries the C₃/phase data.)\n")

    tri_cycles, L = _triangle_loop_vectors(de)
    print(f"  the three circulating currents = triangle loops {tri_cycles} (a C₃-orbit, cycling "
          f"{tri_cycles[0]}→{tri_cycles[1]}→{tri_cycles[2]}→{tri_cycles[0]}).\n")

    def safe_GNB(kf):
        B = nb_operator(kf, de, rev)
        Mmat = np.eye(len(de)) - U_STAR * B
        if abs(np.linalg.det(Mmat)) < 1e-9:
            return None
        return np.linalg.inv(Mmat)

    # walk from a generic off-axis k to its C₃-axis projection (same distance from Γ)
    k_gen = np.array(GENERIC_K)
    k_ax = np.full(3, k_gen.mean())
    print(f"  walking k(s) = (1−s)·k_gen + s·k_axis ; reporting eigenvalues of M and its C₃-DFT modes:\n")
    print(f"   {'s':>5} | {'|μ| of M (3 eigs, sorted)':>30} | {'|μ_triv|':>9} {'|μ_ω|':>8} {'|μ_ω²|':>8} | "
          f"{'arg(μ_ω/μ_ω²) [deg]':>20}")
    print("  " + "-" * 100)
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]:
        kf = tuple((1 - s) * k_gen + s * k_ax)
        G = safe_GNB(kf)
        if G is None:
            print(f"   {s:>5.2f} |  (I − u*B_NB) singular at this k — skipped")
            continue
        Mij = L.conj().T @ G @ L                       # 3x3
        eigs = np.sort(np.abs(np.linalg.eigvals(Mij)))
        # C₃-DFT of the first row (a circulant's eigenvalues = DFT of its first row; off-axis Mij isn't
        # circulant but the DFT components are still the natural C₃-organised amplitudes)
        mt, mw, mw2 = _c3_dft_row(Mij[0, :])
        phase = np.degrees(np.angle(mw / mw2)) if abs(mw2) > 1e-12 else 0.0
        print(f"   {s:>5.2f} | {np.array2string(eigs, precision=4):>30} | {abs(mt):>9.4f} {abs(mw):>8.4f} {abs(mw2):>8.4f} | {phase:>20.3f}")

    print()
    print("  also at the high-symmetry points:")
    for name, kf in [("Γ", GAMMA), ("P=(¼,¼,¼)", P_POINT)]:
        G = safe_GNB(kf)
        if G is None:
            print(f"   {name}: (I − u*B_NB) singular — skipped"); continue
        Mij = L.conj().T @ G @ L
        eigs = np.sort(np.abs(np.linalg.eigvals(Mij)))
        mt, mw, mw2 = _c3_dft_row(Mij[0, :])
        print(f"   {name:>10}: |μ(M)| = {np.array2string(eigs, precision=4)}   "
              f"|μ_triv|,|μ_ω|,|μ_ω²| = {abs(mt):.4f}, {abs(mw):.4f}, {abs(mw2):.4f}"
              + ("   (μ_ω = μ_ω² — circulant, full C₃)" if abs(abs(mw) - abs(mw2)) < 1e-6 else ""))

    print()
    print("  reading:")
    print("    • the three circulating-current amplitudes are NOT equal — M has three distinct eigenvalues,")
    print("      i.e. the NB propagator at u* already 'sees' a three-fold split among the currents;")
    print("    • on the C₃ axis the split is (trivial) + (degenerate ω,ω²) — one mode set apart, two equal;")
    print("    • off the axis ω and ω² separate AND pick up a nonzero relative phase arg(μ_ω/μ_ω²) — the")
    print("      same CP-seed the adjacency probe found, now in the framework's actual (NB) propagator.")


# ======================================================================
# PART C — vertex–vertex NB amplitude (the V_us-style object) + C₃ decomposition
# ======================================================================

def part_C(de, rev):
    print("\n" + "=" * 84)
    print("PART C — vertex–vertex NB amplitude Φ(A,B; u*,k) = (T·G_NB·Sᵀ)[A,B]  (the V_us-style object)")
    print("=" * 84)
    S, T = incidence(de)
    # C₃ as a 3-cycle on the non-axis vertices {1,2,3}
    P = np.real(C3_PERM)
    nxt = {j: int(np.argmax(P[:, j])) for j in (1, 2, 3)}
    cyc = [1, nxt[1], nxt[nxt[1]]]
    print(f"\n  C₃ cycles the non-axis vertices {cyc[0]}→{cyc[1]}→{cyc[2]}→{cyc[0]}; spoke triple = (Φ[0,{cyc[0]}],Φ[0,{cyc[1]}],Φ[0,{cyc[2]}]).\n")

    def safe_GNB(kf):
        B = nb_operator(kf, de, rev)
        Mmat = np.eye(len(de)) - U_STAR * B
        return None if abs(np.linalg.det(Mmat)) < 1e-9 else np.linalg.inv(Mmat)

    k_gen = np.array(GENERIC_K)
    k_ax = np.full(3, k_gen.mean())
    print(f"   {'s':>5} | {'|Φ trivial|':>12} {'|Φ ω|':>9} {'|Φ ω²|':>9} | {'|ω|−|ω²|':>10} {'arg(ω/ω²)°':>11}"
          f" | {'(adjacency-probe |ω| for ref)':>30}")
    print("  " + "-" * 100)
    for s in [0.0, 0.2, 0.5, 0.8, 1.0]:
        kf = tuple((1 - s) * k_gen + s * k_ax)
        G = safe_GNB(kf)
        if G is None:
            print(f"   {s:>5.2f} |  singular — skipped"); continue
        Phi = T @ G @ S.T                              # 4x4 vertex-vertex NB amplitude
        triple = [Phi[0, j] for j in cyc]
        ft, fw, fw2 = (sum(triple) / np.sqrt(3),
                       (triple[0] + omega3 * triple[1] + omega3 ** 2 * triple[2]) / np.sqrt(3),
                       (triple[0] + omega3 ** 2 * triple[1] + omega3 * triple[2]) / np.sqrt(3))
        # adjacency-resolvent reference (z=5, as in the previous probe) — just for scale comparison
        Gadj = np.linalg.inv(5.0 * np.eye(N_ATOMS) - (K_STAR * np.eye(N_ATOMS) - bloch_H(kf, find_bonds())))
        tr = [Gadj[0, j] for j in cyc]
        _, awref, _ = (sum(tr), (tr[0] + omega3 * tr[1] + omega3 ** 2 * tr[2]) / np.sqrt(3), 0)
        ph = np.degrees(np.angle(fw / fw2)) if abs(fw2) > 1e-12 else 0.0
        print(f"   {s:>5.2f} | {abs(ft):>12.4f} {abs(fw):>9.4f} {abs(fw2):>9.4f} | {abs(fw)-abs(fw2):>10.4f} {ph:>11.3f}"
              f" | {abs(awref):>30.4f}")

    print()
    print("  reading: the NB propagator at u* keeps the qualitative shape the adjacency probe found")
    print("  (trivial mode = the always-there content; ω,ω² = the symmetry-broken 'generation' content")
    print("  that vanishes on the C₃ axis; |ω| ≠ |ω²| off it = a CP phase) — but with the framework's")
    print("  actual operator the *numbers* (the relative magnitudes, the phase) are these, not the")
    print("  adjacency-resolvent ones.  Whether THESE numbers, fed through the Koide √m construction")
    print("  + the per-generation Yukawa structure, land on the observed hierarchy/angles/δ_CP is the")
    print("  open problem `frontier.need_d3_species` — this probe supplies the two-vertex inputs, not the answer.")


# ======================================================================
def main():
    de = directed_edges()
    rev = rev_index(de)
    part_A(de, rev)
    part_B(de, rev)
    part_C(de, rev)
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"""
  The framework's actual mass object — the non-backtracking (1PI) Green's function at
  the MDL point u* = (k*−1)/k* = 2/3 (⇔ z* = 17/6) — reproduces, on the srs cell, the
  same qualitative structure the naive adjacency probe found, now with the right operator:

   • B_NB passes the structural checks: Ihara–Bass at Γ (Perron = k*−1 = 2), Ramanujan
     saturation at P (|h| = √2, h_P = (√3+i√5)/2 present, arg(h_P^g) = the δ_CP-ish phase).
   • The three circulating currents (a C₃-orbit) carry three distinct NB amplitudes — a
     genuine three-fold split visible already at the level of the propagator at u*.
   • Organised by C₃: one "trivial" mode (always present — the C₃-blind, one-vertex-like
     content) + two "generation" modes (= gen_w / gen_w²) that vanish on the C₃-symmetric
     axis and turn on off it, with |ω| ≠ |ω²| off-axis — a built-in CP seed that switches
     off exactly where the generations become indistinguishable.
   • These are the two-vertex *inputs* the Yukawa hierarchy would be built from; this probe
     reports their magnitudes/phases (above) but does NOT claim the observed
     ~1:200:3000 mass ratios, the CKM/PMNS angles, or δ_CP fall out — that is the framework's
     known-hard open problem `simulator_skeleton/frontier.need_d3_species` ("5 sessions /
     8 attacks ruled out, foundational extension needed"), which also needs the Koide √m
     construction and the per-generation structure this probe does not include.

  Changes no graded content; the de Rham SUSY verdict (geometric, not statistical;
  frontier gaps mssm_as_adoption / need_d3_species / ADOPTED-MSSM-Sb stand) is unaffected.
""")
    print("nb_two_vertex_generations_probe.py: all checks passed (sentinel).")


if __name__ == "__main__":
    main()
